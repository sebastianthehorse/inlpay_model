from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader

from data.dataset import RaceWindowDataset
from data.streaming_dataset import RaceWindowIterableDataset
from engine.callbacks import EarlyStopping
from models.simple_lstm import SimpleLSTM


class Trainer:
    def __init__(
        self,
        training_features,
        added_features,
        limit_contestants,
        window_timesteps,
        lr: float = 5e-4,
        batch_size: int = 64,
        device: str | torch.device = "cpu",
        stream: bool = False,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.model = SimpleLSTM(
            training_features=training_features,
            added_features=added_features,
            num_contestants=limit_contestants,
        ).to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.window_timesteps = window_timesteps
        self.stream = stream

    def fit(self, self_train_files: Sequence[Path], self_val_files: Sequence[Path], **dataset_kwargs):
        if self.stream:
            train_ds = RaceWindowIterableDataset(
                self_train_files,
                window_timesteps=self.window_timesteps,
                **dataset_kwargs,
            )
        else:
            train_ds = RaceWindowDataset(
                self_train_files,
                window_timesteps=self.window_timesteps,
                **dataset_kwargs,
            )
        val_ds = RaceWindowDataset(
            self_val_files,
            window_timesteps=self.window_timesteps,
            **dataset_kwargs,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=(not self.stream),   # cannot shuffle IterableDataset
            num_workers=2,
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        stopper = EarlyStopping(patience=3, min_delta=0.1, mode="min")
        for epoch in range(1, 100):  # large upper‑bound – we’ll break earlier
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
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
                running += loss.item() * X.size(0)
                batch_size = X.size(0)
                running += loss.item() * batch_size
                n_seen  += batch_size
        return running / n_seen
