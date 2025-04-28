import argparse
import json
import time
from pathlib import Path

import yaml

from engine.evaluator import Evaluator
from engine.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--evaluate", nargs="?", default=False, const=True, help="Run evaluation after training")
    p.add_argument("--stream", nargs="?", default=False, const=True, help="Use streaming IterableDataset during training")
    return p.parse_args()


if __name__ == "__main__":
    start = time.time()
    print("Starting training...")

    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    # Split race files *once* so they stay in the same bucket all run.
    race_files = list(Path(cfg["train_data_dir"]).glob("*.pkl"))
    n_val = int(len(race_files) * cfg["val_split"])
    val_files = race_files[:n_val]
    train_files = race_files[n_val:]

    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    trainer = Trainer(
        training_features=cfg["training_features"],
        added_features=cfg["added_features"],
        limit_contestants=cfg["limit_contestants"],
        window_timesteps=cfg["window_timesteps"],
        lr=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        device=device,
        stream=args.stream,
    )
    trainer.fit(
        self_train_files=train_files,
        self_val_files=val_files,
        training_features=cfg["training_features"],
        target="finishOrder",
        limit_contestants=cfg["limit_contestants"],
    )

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds")

    if args.evaluate:
        test_files = list(Path(cfg["test_data_dir"]).glob("*.pkl"))
        evaluator = Evaluator(trainer.model, device=device)
        metrics = evaluator.run(
            test_files,
            training_features=cfg["training_features"],
            target="finishOrder",
            limit_contestants=cfg["limit_contestants"],
            window_timesteps=cfg["window_timesteps"],
        )
        print("--- Evaluation metrics ---")
        print(json.dumps(metrics, indent=4))
