import numpy as np


class CreateWindowTensor:
    def __init__(self, df, target, n_horses, window_timesteps):
        self.df = df
        self.target = target
        self.n_horses = n_horses
        # self.horse_names = range(1, self.n_horses+1)
        self.window_timesteps = window_timesteps
        self.feature_columns = self._create_feture_columns()
        self.result_array, self.binary_result_array = self._create_result_arrays()

    def _create_feture_columns(self):
        # Prepare feature columns in one go (flattened across contestants and features)
        # return [f"{feature}_{horse}" for horse in self.horse_names for feature in self.features]
        return [col for col in self.df.columns if self.target not in col]

    def _create_result_arrays(self):
        # result_array = self.df[[f'{self.target}_{horse}' for horse in self.horse_names]].iloc[-1].values
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
        num_samples = len(self.df) - self.window_timesteps + 1

        # Preallocate the output array for efficiency
        X = np.empty((num_samples, self.window_timesteps, len(self.feature_columns)), dtype=data_np.dtype)
        y = np.empty((num_samples, self.n_horses), dtype=data_np.dtype)

        # Create sliding windows using numpy slicing
        for i in range(num_samples):
            X[i] = data_np[i : i + self.window_timesteps]
            y[i] = np.array(self.binary_result_array)

        return X, y, self.result_array
