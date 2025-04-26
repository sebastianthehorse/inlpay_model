
class FeatureEngineering:
    def __init__(self, df, training_features, target, limit_contestants, target_int_mapping, target_mapping, n_horses, winner_index):
        self.df = df
        self.training_features = training_features.copy()
        self.target = target
        self.limit_contestants = limit_contestants
        self.target_int_mapping = target_int_mapping
        self.target_mapping = target_mapping
        self.n_horses = n_horses
        self.winner_index = winner_index

    def update_results(self):
        for col in self.target_mapping.keys():
            self.df[col] = self.target_mapping[col]

    def drop_columns(self):
        save = self.training_features + [self.target]
        columns_save = [
            f'{feature}_{i}' for feature in save for i in range(1, self.n_horses + 1)
        ]
        self.df = self.df[columns_save]

    def pick_top_horses(self):
        # Pick the top <limit_contestants> horses by result, from 1 to <limit_contestants>
        save_indexes = [index for index, result in self.target_int_mapping.items()
                        if result <= self.limit_contestants]
        if (len(save_indexes) < self.limit_contestants):
            print('Not enough horses to pick from')
            return False
        if (self.winner_index not in save_indexes):
            print('Winner not in top horses')
            return False
        # Get all columns of the top horses
        save_columns = [column for column in self.df.columns if int(column.split('_')[-1]) in save_indexes]
        self.df = self.df[save_columns]
        return True

    def create_new_features(self):
        # get distance to finish for each horse
        dtf_cols = [col for col in self.df.columns if 'distance_to_finish' in col]
        df_dtf = self.df[dtf_cols]

        # CREATE DISTANCE TO LEADER
        # create new columns for each horse with the distance to the leader
        for col in dtf_cols:
            horse = col.split('_')[-1]
            self.df[f'distance_to_leader_{horse}'] = (self.df[col] - df_dtf.min(axis=1))

        # CREATE LEADER BOOLEAN
        # get the leader at each timestep
        leader_dtf = df_dtf.idxmin(axis=1).values
        # create leader boolean column
        for col in dtf_cols:
            horse = col.split('_')[-1]
            self.df[f'leader_{horse}'] = (leader_dtf == f'distance_to_finish_{horse}').astype(int)

    def prepare_features(self):
        self.update_results()
        self.drop_columns()
        all_good = self.pick_top_horses()
        if not all_good:
            return self.df, False
        self.create_new_features()
        return self.df, True
