from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from data.preprocessing import PreProcessing, DataProcessing
from data.feature_engineering import FeatureEngineering


def build_global_scaler(
    files: List[Path],
    training_features: List[str],
    target: str,
    limit_contestants: int,
) -> Dict[str, Tuple[float, float]]:
    """
    Scan all *training* races once and return {column: (mean, std)}.
    """
    sums, sums2, counts = {}, {}, {}

    for p in files:
        setup = PreProcessing(pd.read_pickle(p), target=target)
        if not setup.valid:
            continue

        df_scaled, _ = DataProcessing(
            df=setup.df,
            winner_index=setup.winner_index,
            training_features=training_features,
        ).process_data()

        dtf_colums = [col for col in df_scaled.columns if "distance_to_finish" in col]
        speed_columns = [col for col in df_scaled.columns if "speed" in col]
        other_features = [col for col in training_features if col not in ["distance_to_finish", "speed"]]
        v_odds_columns = [col for col in df_scaled.columns if col[:-2] in other_features]

        df_dtf = df_scaled[dtf_colums]
        df_speed = df_scaled[speed_columns]
        df_v_odds = df_scaled[v_odds_columns]

        sums["distance_to_finish"] = sums.get("distance_to_finish", 0.0) + df_dtf.values.astype("float64").sum()
        sums2["distance_to_finish"] = sums2.get("distance_to_finish", 0.0) + (df_dtf.values.astype("float64") ** 2).sum()
        counts["distance_to_finish"] = counts.get("distance_to_finish", 0) + len(df_dtf)
        sums["speed"] = sums.get("speed", 0.0) + df_speed.values.sum()
        sums2["speed"] = sums2.get("speed", 0.0) + (df_speed.values ** 2).sum()
        counts["speed"] = counts.get("speed", 0) + len(df_speed)
        sums["v_odds"] = sums.get("v_odds", 0.0) + df_v_odds.values.sum()
        sums2["v_odds"] = sums2.get("v_odds", 0.0) + (df_v_odds.values ** 2).sum()
        counts["v_odds"] = counts.get("v_odds", 0) + len(df_v_odds)

    stats = {}
    for col, n in counts.items():
        mean = sums[col] / n
        var  = (sums2[col] / n) - mean ** 2
        stats[col] = (mean, np.sqrt(max(var, 1e-8)))
    return stats
