import multiprocessing
from pathlib import Path
from typing import Sequence, Callable

import torch
from torch.utils.data import DataLoader

from data.dataset import RaceWindowDataset
from data.streaming_dataset import RaceWindowIterableDataset
from engine.callbacks import EarlyStopping
from models.simple_lstm import SimpleLSTM


class Trainer:
    def __init__(
        self,
        # training_features,
        # added_features,
        # limit_contestants,
        window_timesteps,
        model_factory: Callable[[], torch.nn.Module],
        lr: float = 5e-4,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        stream: bool = False,
        global_stats=None,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size

        # self.model = SimpleLSTM(
        #     training_features=training_features,
        #     added_features=added_features,
        #     num_contestants=limit_contestants,
        # ).to(self.device)
        self.model = model_factory().to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.window_timesteps = window_timesteps
        self.stream = stream
        self.global_stats = global_stats

    def fit(self, train_files: Sequence[Path], val_files: Sequence[Path], **dataset_kwargs):
        if self.stream:
            train_data_streamer = RaceWindowIterableDataset(
                train_files,
                window_timesteps=self.window_timesteps,
                global_stats=self.global_stats,
                **dataset_kwargs,
            )
        else:
            train_data_streamer = RaceWindowDataset(
                train_files,
                window_timesteps=self.window_timesteps,
                global_stats=self.global_stats,
                **dataset_kwargs,
            )
        val_data_streamer = RaceWindowDataset(
            val_files,
            window_timesteps=self.window_timesteps,
            global_stats=self.global_stats,
            **dataset_kwargs,
        )
        train_data_loader = DataLoader(
            train_data_streamer,
            batch_size=self.batch_size,
            shuffle=(not self.stream),  # cannot shuffle IterableDataset
            num_workers=max(1, multiprocessing.cpu_count() // 2),
        )
        val_data_loader = DataLoader(val_data_streamer, batch_size=self.batch_size)

        stopper = EarlyStopping(patience=3, min_delta=0.1, mode="min")

        for epoch in range(1, 100):  # early stopping will break the loop
            train_loss = self._run_epoch(train_data_loader, train=True)
            val_loss = self._run_epoch(val_data_loader, train=False)
            print(f"Epoch {epoch:3d} • train={train_loss:.3f} • val={val_loss:.3f}")
            if stopper.step(val_loss):
                print("Early stopping triggered → best val loss", f"{stopper.best_score:.3f}")
                break

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(mode=train)
        running, n_seen = 0.0, 0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                out = self.model(X)
                loss = self.loss_fn(out, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                batch_size = X.size(0)
                running += loss.item() * batch_size
                n_seen += batch_size
        return running / n_seen
