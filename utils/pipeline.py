import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .data import preprocess, build_mapping
from sklearn.model_selection import train_test_split


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def invmape(y_true, y_pred):
    return 100. - mape(y_true, y_pred)


def get_data(train, base, test=None, cv_ratio=None):
    y = train['avg_price_sqm']
    val = None
    if cv_ratio is not None:
        train, val, yt, yv = train_test_split(train, y, test_size=cv_ratio)

    mapping = build_mapping(train, base)
    train = preprocess(train, mapping)
    res = [(train, yt)]
    if val is not None:
        val = preprocess(val, mapping)
        res.append((val, yv))
    if test is not None:
        test = preprocess(test, mapping)
        res.append(test)
    return res


def pipeline(model, train, base, test=None, cv_ratio=None):
    data = get_data(train, base, test, cv_ratio)
    print('Data processed')
    if cv_ratio is None:
        val = None
    else:
        val = data[1]
    model.fit(data[0][0], data[0][1], val)
    metric = invmape(data[0][1], model.predict(data[0][0]))
    print(f'Метрика на train: {metric}')

    if cv_ratio is not None:
        val_pred = model.predict(data[1][0])
        metric = invmape(data[1][1], val_pred)
        print(f'Метрика на валидации: {metric}')

    artifacts = {'data': data, 'model': model,
                 'importances': sorted(zip(model.model.feature_importances_,
                                           list(data[0][0].columns)), reverse=True)}

    if test is not None:
        pred = model.predict(data[-1])
        return pred, artifacts

    return artifacts
