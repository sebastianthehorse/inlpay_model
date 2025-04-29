from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.feature_engineering import FeatureEngineering
from data.preprocessing import DataProcessing, PreProcessing
from data.window import CreateWindowTensor


class RaceWindowDataset(Dataset):
    """Generates (window, label) tensors from a *list* of racetrack pickle files.

    Each `__getitem__` returns one sliding window (shape: `(timesteps, n_feats)`) and
    its winner index label (int).
    """

    def __init__(
        self,
        files: Sequence[Path],
        training_features: List[str],
        target: str,
        limit_contestants: int,
        window_timesteps: int,
        randomize: bool = False,
    ):
        self.X, self.y = self._combine_races_to_tensor(
            files,
            training_features,
            target,
            limit_contestants,
            window_timesteps,
            randomize,
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.as_tensor(self.X[idx], dtype=torch.float32),
            torch.as_tensor(self.y[idx], dtype=torch.long),
        )

    @staticmethod
    def _load_and_validate(pkl_file: Path, target: str) -> Optional[PreProcessing]:
        """Load a pickle file and run the initial validation.

        Returns a *PreProcessing* object if the race passes sanity checks;
        otherwise returns **None** so the caller can skip the file.
        """
        df_race = pd.read_pickle(pkl_file)
        setup = PreProcessing(df=df_race, target=target)
        if not setup.valid:
            return None
        return setup

    @staticmethod
    def _make_feature_df(
        setup: PreProcessing,
        training_features: List[str],
        target: str,
        limit_contestants: int,
    ) -> pd.DataFrame:
        """Run scaling + feature engineering and return the feature DataFrame."""
        data_prep = DataProcessing(
            df=setup.df,
            winner_index=setup.winner_index,
            training_features=training_features,
        )
        df_scaled, _ = data_prep.process_data()

        feat_prep = FeatureEngineering(
            df=df_scaled,
            training_features=training_features,
            target=target,
            limit_contestants=limit_contestants,
            target_int_mapping=setup.target_int_mapping,
            target_mapping=setup.target_mapping,
            n_horses=setup.n_horses,
            winner_index=setup.winner_index,
        )
        return feat_prep.prepare_features()

    # Internal utility used only by this eager‑loading dataset
    @staticmethod
    def _combine_races_to_tensor(
        pkl_files: Sequence[Path],
        training_features: List[str],
        target: str,
        limit_contestants: int,
        window_timesteps: int,
        randomize: bool,
    ):
        X_list, y_list = [], []

        for pkl_file in pkl_files:
            setup = RaceWindowDataset._load_and_validate(pkl_file, target)
            if setup is None:
                continue  # skip invalid race

            try:
                df_feature = RaceWindowDataset._make_feature_df(
                    setup,
                    training_features=training_features,
                    target=target,
                    limit_contestants=limit_contestants,
                )
            except ValueError as e:
                print(f"!!## BROKEN DF: {pkl_file} !!## STATUS: {e}")
                continue

            if len(df_feature) < window_timesteps:
                continue  # race too short → skip

            windower = CreateWindowTensor(
                df=df_feature,
                target=target,
                n_horses=limit_contestants,
                window_timesteps=window_timesteps,
            )
            X, y_onehot, _ = windower.create_sliding_windows2()
            X_list.append(X)
            y_list.append(np.argmax(y_onehot, axis=1))  # → single‑class label

        if not X_list:
            raise RuntimeError("No valid races found while building dataset. Check paths and preprocessing steps.")

        X = np.concatenate(X_list)
        y = np.concatenate(y_list)

        if randomize:
            rng = np.random.default_rng(seed=42)
            perm = rng.permutation(len(X))
            X, y = X[perm], y[perm]
        return X, y
