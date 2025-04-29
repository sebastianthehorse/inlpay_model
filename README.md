# Code Format and Style

Run these commands before creating a PR:

- `ruff format` (Python formatter)
- `ruff check --fix` (Python linter)
- `pyright` (Python static type checker)

Or as a one-liner: `ruff format && ruff check --fix && pyright`

# Train CLI

Example
```bash
python train.py --config configs/baseline.yaml --evaluate --stream
```
