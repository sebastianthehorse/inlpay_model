import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, training_features, added_features, num_contestants):
        super(SimpleLSTM, self).__init__()
        
        num_features = len(training_features + added_features)
        self.num_contestants = num_contestants
        input_size = num_contestants * num_features

        # Conv1D expects (batch_size, channels, seq_len)
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=32)
        self.pool = nn.AvgPool1d(kernel_size=2)

        # LSTM after conv/pool
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)

        # Stronger dropout
        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(32, 64)
        
        # Final output layer: logits for each contestant
        self.output_layer = nn.Linear(64, num_contestants)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        # Conv1D needs (batch_size, features, seq_len)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = self.pool(x)
        # Prepare for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = F.relu(self.dense1(x))

        logits = self.output_layer(x)

        return logits
