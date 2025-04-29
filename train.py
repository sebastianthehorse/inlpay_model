import argparse
import time
from pathlib import Path
from datetime import datetime

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

    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    race_files = list(Path(cfg["train_data_dir"]).glob("*.pkl"))
    n_val = int(len(race_files) * cfg["val_split"])
    val_files = race_files[:n_val]
    train_files = race_files[n_val:]

    print(f"Starting training on {len(train_files)} files, validating on {len(val_files)} files")

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
        train_files=train_files,
        val_files=val_files,
        training_features=cfg["training_features"],
        target=cfg["target"],
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
            target=cfg["target"],
            limit_contestants=cfg["limit_contestants"],
            window_timesteps=cfg["window_timesteps"],
        )
        # dump evaluation summary to versioned results csv file
        results_dir = Path(cfg["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(results_file, "w") as f:
            f.write("winner_accuracy,val_loss,avg_precision,brier_score,log_loss,roc_curve,pr_curve\n")
            f.write(
                f"{metrics['winner_accuracy']},{metrics['val_loss']},{metrics['avg_precision']},{metrics['brier_score']},{metrics['log_loss']}\n"
            )
        print(f"Evaluation results saved to {results_file}")
