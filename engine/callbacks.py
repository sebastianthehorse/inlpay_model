class EarlyStopping:
    """Stop training when the monitored metric has stopped improving."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, current: float) -> bool:
        if self.best_score is None:
            self.best_score = current
            return False

        improve = (
            current < self.best_score - self.min_delta
            if self.mode == "min"
            else current > self.best_score + self.min_delta
        )

        if improve:
            self.best_score = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
