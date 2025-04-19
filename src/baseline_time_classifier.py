import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class BaselineTimeClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple rule-based classifier that predicts 1 if the time is between start and end
    and the day is a weekend, otherwise predicts 0.
    """

    def __init__(self, start="07:00", end="19:00"):
        self.start = start
        self.end = end

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame must have a DatetimeIndex")

        df = X.copy()
        start_time = pd.to_datetime(self.start).time()
        end_time = pd.to_datetime(self.end).time()

        df["y_pred"] = (df.index.time >= start_time) & (df.index.time <= end_time)
        df["y_pred"] &= df.index.weekday < 5
        df["y_pred"] = df["y_pred"].astype(int)

        return df["y_pred"].values

    def score(self, X, y):
        """Computes accuracy score for evaluation."""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
