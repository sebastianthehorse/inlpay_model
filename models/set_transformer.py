import torch.nn as nn


class SetTransformer(nn.Module):
    """
    Permutation-invariant network:
      shared MLP per horse
      multi-head self-attention across horses
      max-pool -> dense -> softmax
    """

    def __init__(self, training_features, added_features, num_contestants):
        super().__init__()
        self.n_feats = len(training_features + added_features)

        # per-horse sub-net (shared weights)
        self.horse_mlp = nn.Sequential(
            nn.Linear(self.n_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # Self-attention across horses (timesteps collapsed)
        self.attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
            num_layers=2,
        )

        # classification
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_contestants),
        )

    def forward(self, x):
        # x: (batch, timesteps, horses*features)
        B, T, _ = x.shape
        H = self.horse_mlp  # alias

        # reshape -> (batch, timesteps, horses, feats)
        horses = x.view(B, T, -1, self.n_feats)

        # shared MLP per horse
        horses = H(horses)

        # collapse time dimension by max-pooling
        horses, _ = horses.max(dim=1)        # (batch, horses, 64)

        # self-attention across horses
        horses = self.attn(horses)           # (batch, horses, 64)

        # global max-pool over horses
        pooled, _ = horses.max(dim=1)        # (batch, 64)

        return self.classifier(pooled)       # (batch, n_contestants)
