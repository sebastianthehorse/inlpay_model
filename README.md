# TODO
* Think about scaling each race individually: It will hide differences between races
* Try mini batching single races
* Consider whether the binary_result_array logic (1 if result == 1, else 0): if we want to predict all ranks, switch to a multi-class label array via the existing result_array

# Code

Run these commands before creating a PR:

- `ruff format` (Python formatter)
- `ruff check --fix` (Python linter)
- `pyright` (Python static type checker)

Or as a one-liner: `ruff format && ruff check --fix && pyright`
