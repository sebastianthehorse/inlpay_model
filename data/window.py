import numpy as np
import pandas as pd


class CreateWindowTensor:
    def __init__(self, df: pd.DataFrame, target: str,
                 n_horses: int, window_timesteps: int,
                 stride: int = 1):
        self.df = df
        self.target = target
        self.n_horses = n_horses
        self.window_timesteps = window_timesteps
        self.feature_columns = self._create_feture_columns()
        self.result_array, self.binary_result_array = self._create_result_arrays()
        self.stride = self._set_stride(stride)

    def _set_stride(self, stride: int):
        if stride == 9999:  # Only train on the last window
            return len(self.df) - self.window_timesteps + 1
        return stride

    def _create_feture_columns(self):
        # Prepare feature columns in one go (flattened across contestants and features)
        # return [f"{feature}_{horse}" for horse in self.horse_names for feature in self.features]
        return [col for col in self.df.columns if self.target not in col]

    def _create_result_arrays(self):
        result_array = self.df[[col for col in self.df.columns if self.target in col]].iloc[-1].values
        binary_result_array = []
        for result in result_array:
            if result == 1:
                binary_result_array.append(1)
            else:
                binary_result_array.append(0)
        return result_array, binary_result_array

    def create_sliding_windows2(self):
        # Convert to numpy for efficient slicing
        data_np = self.df[self.feature_columns].values

        # Window size and stride
        num_samples_total = len(self.df) - self.window_timesteps + 1
        keep_idx = range(0, num_samples_total, self.stride)
        num_samples = len(keep_idx)

        # Preallocate the output array for efficiency
        X = np.empty((num_samples, self.window_timesteps, len(self.feature_columns)), data_np.dtype)
        y = np.empty((num_samples, self.n_horses), dtype=data_np.dtype)

        # Create sliding windows using numpy slicing
        for j, i in enumerate(keep_idx):
            X[j] = data_np[i : i + self.window_timesteps]
            y[j] = self.binary_result_array

        return X, y, self.result_array
