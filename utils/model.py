import xgboost
import pandas as pd
import numpy as np


def xgbmape(y_pred, y_true):
    gt = y_true.get_label()
    return 'mape', np.mean(np.abs((gt - y_pred) / gt)) * 100


class XGBModel:
    def __init__(self, params):
        self.model = xgboost.XGBRegressor(**params)

    def fit(self, X, y, val):
        if type(X) == pd.DataFrame:
            X = X.values
        if type(y) == pd.Series:
            y = y.values

        val_x, val_y = val
        if type(val_x) == pd.DataFrame:
            val_x = val_x.values
        if type(val_y) == pd.Series:
            val_y = val_y.values

        print(X.shape, y.shape, val_x.shape, val_y.shape)

        if val is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, eval_set=[(X, y), (val_x, val_y)], eval_metric=xgbmape)

    def predict(self, X):
        if type(X) == pd.DataFrame:
            X = X.values

        return self.model.predict(X)
