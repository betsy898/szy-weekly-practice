"""固定预测规则、滞后线性回归及 Ridge 候选模型。"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def predict_last_value(x):
    return np.asarray(x)[:, -1]


def predict_window_mean(x):
    return np.asarray(x).mean(axis=1)


def fit_lag_linear(x_train, y_train):
    model = make_pipeline(StandardScaler(), LinearRegression())
    return model.fit(x_train, y_train)


def fit_ridge(x_train, y_train, alpha):
    if alpha <= 0:
        raise ValueError("Ridge alpha 必须为正数")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    return model.fit(x_train, y_train)
