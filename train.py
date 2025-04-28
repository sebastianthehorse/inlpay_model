"""Command line entry point

Examples:
  # Train only
  python -m inlpay_model.train --config configs/baseline.yaml

  # Train then evaluate on held out test folder
  python -m inlpay_model.train --config configs/baseline.yaml --evaluate --test_dir ~/Projects/data/hts/test
"""

from pathlib import Path
import argparse
import yaml
import json

from engine.trainer import Trainer
from engine.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument('--evaluate', nargs="?", default=False, const=True, help="Run evaluation after training")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    # Split race files *once* so they stay in the same bucket all run.
    race_files = list(Path(cfg["train_data_dir"]).glob("*.pkl"))
    n_val = int(len(race_files) * cfg["val_split"])
    val_files = race_files[:n_val]
    train_files = race_files[n_val:]

    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"

    trainer = Trainer(
        training_features=cfg["training_features"],
        added_features=cfg["added_features"],
        limit_contestants=cfg["limit_contestants"],
        window_timesteps=cfg["window_timesteps"],
        lr=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        device=device,
    )
    trainer.fit(
        self_train_files=train_files,
        self_val_files=val_files,
        training_features=cfg["training_features"],
        target="finishOrder",
        limit_contestants=cfg["limit_contestants"],
        randomize=False,
    )
    if args.evaluate:
        if args.test_dir is None:
            raise SystemExit("--test_dir must be supplied when --evaluate is set")

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
