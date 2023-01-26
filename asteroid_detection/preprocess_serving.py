from typing import Any

import pandas as pd
import numpy as np
import xgboost as xgb


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        pass

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        df = pd.DataFrame(columns=body.keys())
        df.loc[0] = body.values()
        df['avg_dia'] = df[['Est Dia in KM(min)', 'Est Dia in KM(max)']].mean(axis=1)
        X = df[['Absolute Magnitude', 'avg_dia', 'Relative Velocity km per hr', 'Miss Dist.(kilometers)', 'Orbit Uncertainity',
                'Minimum Orbit Intersection', 'Jupiter Tisserand Invariant', 'Epoch Osculation', 'Eccentricity', 'Semi Major Axis',
                'Inclination', 'Asc Node Longitude', 'Orbital Period', 'Perihelion Distance', 'Perihelion Arg',
                'Aphelion Dist', 'Perihelion Time', 'Mean Anomaly', 'Mean Motion']]
        # we expect to get four valid numbers on the dict: x0, x1, x2, x3
        return xgb.DMatrix(X)

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=round(data[0]), y_raw=float(data[0]))
