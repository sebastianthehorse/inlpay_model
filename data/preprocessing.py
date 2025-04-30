import pandas as pd


class PreProcessing:
    DQ_MAPPING = {31: 1, 32: 2, 33: 3, 34: 4, 35: 5, 36: 6, 37: 7, 38: 8, 39: 9, 40: 10, 41: 11, 42: 12, 43: 13, 44: 14, 45: 15}

    def __init__(self, df, target):
        self.df = df
        self.target = target
        self._drop_missing_data()
        self.n_horses = self._get_n_horses()
        self.target_mapping, self.target_int_mapping = self._get_target_mappings()
        self.winner_index = self._get_winner_index()
        self.valid = self._check_valid()

    def _drop_missing_data(self):
        self.df = self.df.dropna().reset_index(drop=True)

    def _get_n_horses(self):
        n_horses = len([col for col in self.df.columns if self.target in col])
        return n_horses

    def _get_target_mappings(self):
        result = {f"{self.target}_{i}": int(self.df[f"{self.target}_{i}"].iloc[-1]) for i in range(1, self.n_horses + 1)}
        # first map the DQ results
        target_mapping = {}
        for key, value in result.items():
            if value in self.DQ_MAPPING.keys():
                target_mapping[key] = self.DQ_MAPPING[value]
        # then map the normal results
        sorted_result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
        for key, value in sorted_result.items():
            updated_value = value
            if key in target_mapping.keys():
                continue
            while True:
                if updated_value not in target_mapping.values():
                    break
                updated_value += 1
            target_mapping[key] = updated_value
        target_int_mapping = {int(k.split("_")[-1]): v for k, v in target_mapping.items()}
        return target_mapping, target_int_mapping

    def _get_winner_index(self):
        winner_column = self.df[[f"winner_{i}" for i in range(1, self.n_horses + 1)]].loc[0].idxmax()
        winner_index = int(winner_column.split("_")[-1])
        return winner_index

    def _check_valid(self):
        if not self.n_horses or not self.winner_index:
            return False
        return True


class DataProcessing:
    training_features_required = ["distance_to_finish"]

    def __init__(self, df, winner_index, training_features, global_stats: dict[str, tuple[float, float]] | None = None):
        self.df = df
        self.winner_index = winner_index
        self.training_features = training_features
        self.global_stats = global_stats

    def drop_tail(self):
        if self.winner_index:
            self.df = self.df[self.df[f"distance_to_finish_{self.winner_index}"] > 0]
        else:
            self.df = self.df[self.df["distance_to_finish_1"] > 0]

    def run_on_second_half(self):
        self.df = self.df.iloc[len(self.df) // 2 :]

    def convert_epoch_to_datetime_and_set_index(self):
        self.df["epoch"] = pd.to_datetime(self.df["epoch"], origin="unix", unit="ns")
        self.df = self.df.set_index("epoch")

    def downsample_data(self):
        self.df = self.df.iloc[::2]

    def scale_data(self):
        df_scaled = self.df.copy()
        # Separate the different feature columns
        dtf_colums = [col for col in df_scaled.columns if "distance_to_finish" in col]
        speed_columns = [col for col in df_scaled.columns if "speed" in col]
        other_features = [col for col in self.training_features if col not in ["distance_to_finish", "speed"]]
        v_odds_columns = [col for col in df_scaled.columns if col[:-2] in other_features]
        # Separate dataframes for dtf, speed and v_odds
        df_dtf = df_scaled[dtf_colums]
        df_speed = df_scaled[speed_columns]
        df_v_odds = df_scaled[v_odds_columns]
        df_rest = df_scaled.drop(columns=dtf_colums + speed_columns + v_odds_columns, axis=1)
        # std scaling
        if self.global_stats is None:
            df_dtf_scaled = (df_dtf - df_dtf.values.mean()) / df_dtf.values.std(ddof=1)
            df_speed_scaled = (df_speed - df_speed.values.mean()) / df_speed.values.std(ddof=1)
            df_v_odds_scaled = (df_v_odds - df_v_odds.values.mean()) / df_v_odds.values.std(ddof=1)
        else:
            df_dtf_scaled = (df_dtf - self.global_stats["distance_to_finish"][0]) / self.global_stats["distance_to_finish"][1]
            df_speed_scaled = (df_speed - self.global_stats["speed"][0]) / self.global_stats["speed"][1]
            df_v_odds_scaled = (df_v_odds - self.global_stats["v_odds"][0]) / self.global_stats["v_odds"][1]

        # combine the scaled dataframes and the rest of the data
        self.df_scaled = pd.concat([df_dtf_scaled, df_speed_scaled, df_v_odds_scaled, df_rest], axis=1)
        self.df_scaled.reindex(sorted(df_scaled.columns), axis=1)

    def round_df(self):
        self.df_scaled = self.df_scaled.round(2)

    def process_data(self):
        self.drop_tail()
        self.run_on_second_half()
        self.downsample_data()
        self.scale_data()
        self.round_df()
        # self.convert_epoch_to_datetime_and_set_index()
        return self.df_scaled, self.df
