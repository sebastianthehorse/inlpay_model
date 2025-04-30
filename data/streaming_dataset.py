from pathlib import Path
from random import Random
from typing import List, Sequence, Iterator, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from data.dataset import RaceWindowDataset
from data.window import CreateWindowTensor


class RaceWindowIterableDataset(IterableDataset):
    """Stream windows one-by-one so memory stays (roughly) constant.

    Each worker process gets its own slice of the race-file list
    to avoid duplicate work when num_workers > 0.
    """

    def __init__(
        self,
        files: Sequence[Path],
        training_features: List[str],
        target: str,
        limit_contestants: int,
        window_timesteps: int,
        shuffle_files: bool = True,
        seed: int = 42,
        window_stride: int = 1,
        global_stats=None,
    ) -> None:
        self.files = list(files)
        self.training_features = training_features
        self.target = target
        self.limit_contestants = limit_contestants
        self.window_timesteps = window_timesteps
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.window_stride = window_stride
        self.global_stats = global_stats

    def _get_shard(self, worker_id: int, num_workers: int) -> List[Path]:
        """Split the file list so each DataLoader worker gets unique races."""
        return self.files[worker_id::num_workers]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process DataLoader
            shard = self.files
            rng = Random(self.seed)
        else:
            shard = self._get_shard(worker_info.id, worker_info.num_workers)
            rng = Random(self.seed + worker_info.id)

        while True:  # one full pass is an 'epoch'
            if self.shuffle_files:
                rng.shuffle(shard)

            for pkl_file in shard:
                df_race = RaceWindowDataset._load_and_validate(  # reuse helper
                    pkl_file, self.target
                )
                if df_race is None:
                    continue

                try:
                    df_feature = RaceWindowDataset._make_feature_df(
                        df_race,
                        training_features=self.training_features,
                        target=self.target,
                        limit_contestants=self.limit_contestants,
                        global_stats=self.global_stats,
                    )
                    if len(df_feature) < self.window_timesteps:
                        continue
                except ValueError as e:
                    print(f"!!## BROKEN DF: {pkl_file}!!## STATUS: {e}")
                    continue

                windower = CreateWindowTensor(
                    df=df_feature,
                    target=self.target,
                    n_horses=self.limit_contestants,
                    window_timesteps=self.window_timesteps,
                    stride=self.window_stride,
                )
                X, y_onehot, _ = windower.create_sliding_windows2()
                y = np.argmax(y_onehot, axis=1)  # single-class label

                for Xi, yi in zip(X, y):
                    yield torch.as_tensor(Xi, dtype=torch.float32), torch.as_tensor(yi, dtype=torch.long)

            return
