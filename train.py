import argparse
import time
from datetime import datetime
from pathlib import Path

import yaml

from engine.evaluator import Evaluator
from engine.trainer import Trainer
from data.global_scaler import build_global_scaler

from models.simple_lstm import SimpleLSTM
from models.set_transformer import SetTransformer


MODEL_ZOO = {
    "lstm": SimpleLSTM,
    "set_transformer": SetTransformer,
}


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

    print("Fitting global feature scaler …")
    global_stats = build_global_scaler(
        train_files,
        training_features=cfg["training_features"],
        target="finishOrder",
        limit_contestants=cfg["limit_contestants"],
    )

    print(f"Starting training on {len(train_files)} files, validating on {len(val_files)} files")

    device = "mps" if __import__("torch").backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    cfg_model = cfg.get("model", "lstm")
    trainer = Trainer(
        model_factory=lambda: MODEL_ZOO[cfg_model](
            training_features=cfg["training_features"],
            added_features=cfg["added_features"],
            num_contestants=cfg["limit_contestants"],
        ),
        window_timesteps=cfg["window_timesteps"],
        lr=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        device=device,
        stream=args.stream,
        global_stats=global_stats,
    )
    trainer.fit(
        train_files=train_files,
        val_files=val_files,
        training_features=cfg["training_features"],
        target=cfg["target"],
        limit_contestants=cfg["limit_contestants"],
        window_stride=cfg["window_stride"],
    )

    train_time = time.time() - start
    print(f"Training completed in {train_time:.2f} seconds")

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
            f.write("model,winner_accuracy,val_loss,avg_precision,brier_score,log_loss,time\n")
            f.write(f"{cfg_model},{metrics['winner_accuracy']},{metrics['val_loss']},{metrics['avg_precision']},{metrics['brier_score']},{metrics['log_loss']},{train_time}\n")
        print(f"Evaluation results saved to {results_file}")
