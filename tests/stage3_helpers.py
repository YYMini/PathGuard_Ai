from __future__ import annotations
import numpy as np
import pandas as pd
from src.feature_engineering import generate_trajectory_features
from src.data_quality import add_quality_flags

def trajectory(identifier="t1", points=120, offset=0.0):
    raw = pd.DataFrame({"user_id": ["000"] * points, "trajectory_id": [identifier] * points,
        "timestamp": pd.date_range("2024-01-01", periods=points, freq="10s"),
        "latitude": 39.0 + offset + np.arange(points) * .0001,
        "longitude": 116.0 + np.arange(points) * .0001, "altitude": [10.0] * points})
    return generate_trajectory_features(raw)[0]

def checked_many(count=5, points=120):
    return add_quality_flags(pd.concat([trajectory(f"t{i}", points, i * .01) for i in range(count)], ignore_index=True))
