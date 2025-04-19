import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """select specific columns of a given dataset"""

    def __init__(self, feature_selection=None):
        if not isinstance(feature_selection, list) and feature_selection is not None:
            raise
        elif isinstance(feature_selection, list):
            self.feature_selection = feature_selection
        else:
            self.feature_selection = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.feature_selection:
            X = X.loc[:, self.feature_selection]

        if isinstance(X, pd.Series):
            X = X.to_frame()

        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to specific columns in a DataFrame."""

    def __init__(self, columns=None, add_constant=1.0):
        if not isinstance(columns, list) and columns is not None:
            raise ValueError("columns must be a list or None")
        self.columns = columns
        self.add_constant = add_constant

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        if self.columns is None:
            return X

        # for col in self.columns:
        #     if col not in X.columns:
        #         raise ValueError(f"Column '{col}' not in DataFrame.")
        #     if (X[col] + self.add_constant <= 0).any():
        #         raise ValueError(
        #             f"Column '{col}' contains values <= -{self.add_constant}, which cannot be log-transformed.")
        #     X[col] = np.log(X[col] + self.add_constant)

        # Check for columns that contain strings from self.columns
        for col in X.columns:
            if any(c in col for c in self.columns):
                if (X[col] + self.add_constant <= 0).any():
                    raise ValueError(
                        f"Column '{col}' contains values <= -{self.add_constant}, which cannot be log-transformed.")
                    continue
                X[col] = np.log(X[col] + self.add_constant)
        return X


class CustomTimeSeriesSplitter:
    """Custom time series cross-validation splitter."""

    def __init__(self, cv_splits, df_index):
        self.cv_splits = cv_splits
        self.df_index = df_index

    def split(self, X, y=None, groups=None):
        for split_name, split_config in self.cv_splits.items():
            train_dates = pd.to_datetime(split_config['train_dates'])
            val_dates = pd.to_datetime(split_config['val_dates'])

            # Create masks for training and validation
            train_mask = self.df_index.floor('D').isin(train_dates)
            val_mask = self.df_index.floor('D').isin(val_dates)

            # Get indices
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]

            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.cv_splits)


class AddHistoricOfficeOccupancyFeature(BaseEstimator, TransformerMixin):
    """
    Transformer to add historical office occupancy features based on time patterns.

    This transformer creates a new feature representing the average occupancy
    for specific time periods across weekdays, capturing typical office usage patterns.

    Parameters:
    -----------
    add_feature : bool, default=True
        Whether to add the historical occupancy feature
    period_length : int, default=15
        Length of time period in minutes, must divide a day (1440 minutes) without remainder
    """

    def __init__(self, add_feature=True, period_length=15):
        self.add_feature = add_feature
        self.period_length = period_length
        self.period_averages_ = None
        self._validate_period_length()

    def _validate_period_length(self):
        """Ensure period_length divides a day evenly"""
        minutes_in_day = 24 * 60
        if minutes_in_day % self.period_length != 0:
            raise ValueError(f"period_length ({self.period_length}) must divide a day (1440 minutes) evenly")

    def fit(self, X, y):
        """
        Calculate average occupancy for each time period across weekdays.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features with datetime index
        y : pandas.Series or numpy.ndarray
            Target occupancy values

        Returns:
        --------
        self : object
        """
        if not self.add_feature:
            return self

        if isinstance(X, pd.DataFrame) and isinstance(X.index, pd.DatetimeIndex):
            dt_index = X.index
        else:
            raise ValueError("X must have a datetime index")

        if not isinstance(y, pd.Series):
            y_series = pd.Series(y, index=dt_index)
        else:
            y_series = y.copy()

        data = pd.DataFrame({'occupancy': y_series}, index=dt_index)
        data['is_weekday'] = data.index.weekday < 5

        minutes_since_midnight = data.index.hour * 60 + data.index.minute
        data['time_period'] = minutes_since_midnight // self.period_length

        period_averages = data[data['is_weekday']].groupby('time_period')['occupancy'].mean()

        self.period_averages_ = period_averages.to_dict()

        return self

    def transform(self, X):
        """
        Add historical occupancy feature based on time patterns.

        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Input features with datetime index

        Returns:
        --------
        X_new : pandas.DataFrame or numpy.ndarray
            Original features with additional historical occupancy feature
        """
        if not self.add_feature:
            return X

        df = X.copy()
        is_numpy = False

        if isinstance(df, pd.DataFrame) and isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index
        else:
            raise ValueError("X must have a datetime index")

        is_weekday = dt_index.weekday < 5

        minutes_since_midnight = dt_index.hour * 60 + dt_index.minute
        time_period = minutes_since_midnight // self.period_length

        df['historic_occupancy'] = 0.0

        for period, avg_value in self.period_averages_.items():
            mask = (time_period == period) & is_weekday
            df.loc[mask, 'historic_occupancy'] = avg_value

        return df


def get_pipeline(model=LogisticRegression,
                 scaler=StandardScaler,
                 historic_feature=AddHistoricOfficeOccupancyFeature,
                 feature_selector=ColumnSelector,
                 log_transformer=LogTransformer,
                 scaler_params=None,
                 model_params=None,
                 historic_feature_params=None,
                 feature_selector_params=None,
                 log_transformer_params=None
                 ):
    """
    Create a scikit-learn pipeline with customizable scaler and model parameters.

    Returns:
    --------
    sklearn.pipeline.Pipeline
        Configured pipeline with scaler and model
    """
    if log_transformer_params is None:
        log_transformer_params = {"columns": ['Light', 'CO2']}

    if scaler_params is None:
        scaler_params = {}

    if model_params is None:
        model_params = {"random_state": 42, "class_weight": "balanced"}

    if historic_feature_params is None:
        historic_feature_params = {"period_length": 30}

    if feature_selector_params is None:
        feature_selector_params = {"feature_selection": ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']}

    pipeline = Pipeline([
        ("log_transformer", log_transformer(**log_transformer_params)),
        ("feature_selector", feature_selector(**feature_selector_params)),
        ("historic_feature", historic_feature(**historic_feature_params)),
        ("scaler", scaler(**scaler_params)),
        ("model", model(**model_params))
    ])

    return pipeline
