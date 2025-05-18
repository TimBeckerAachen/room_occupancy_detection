import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

from .plotting import plot_misclassifications


features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
target = "Occupancy"

partial_days = [
                '2015-02-04',
                '2015-02-10',
                '2015-02-11',
                '2015-02-18',
                '2015-02-02',
]

full_days = [
            '2015-02-06',
            '2015-02-09',
            '2015-02-12',
            '2015-02-13',
            '2015-02-16',
            '2015-02-17',
            '2015-02-03',
            '2015-02-05',
]

weekends = [
    '2015-02-07',
    '2015-02-08',
    '2015-02-14',
    '2015-02-15',
]

# no validation dataset excluded and more training data
cv_splits_large = {
    'first_split': {
        'train_dates': partial_days[:3] + weekends[0:2] + full_days[:4] + full_days[-2:] + weekends[-1:] + partial_days[-1:],
        'val_dates': [weekends[2]] + full_days[4:-2] + [partial_days[3]]
    },
    'second_split': {
        'train_dates': partial_days[1:4] + weekends[1:3] + full_days[2:-2] + full_days[-2:] + weekends[-1:] + partial_days[-1:],
        'val_dates': [weekends[0]] + full_days[:2] + [partial_days[0]]
    },
    'third_split': {
        'train_dates': [partial_days[0]] + partial_days[2:4] + [weekends[0]] + weekends[2:3] + full_days[:2] + full_days[4:-2] + full_days[-2:] + weekends[-1:] + partial_days[-1:],
        'val_dates': [weekends[1]] + full_days[2:4] + [partial_days[1]]
    },
    'fourth_split': {
        'train_dates': partial_days[0:2] + partial_days[3:4] + weekends[0:2] + full_days[:4] + full_days[6:-2] + full_days[-2:] + weekends[-1:] + partial_days[-1:],
        'val_dates': [weekends[2]] + full_days[4:6] + [partial_days[2]]
    },
    'fifth_split': {
        'train_dates': partial_days[0:3] + weekends[:3] + full_days[1:4] + full_days[5:-3] + full_days[-2:] + weekends[-1:] + partial_days[-1:],
        'val_dates': [full_days[0], full_days[4], full_days[-3]] + [partial_days[3]]
    },
}

# validation dataset excluded
cv_splits_no_val = {
    'first_split': {
        'train_dates': partial_days[:3] + weekends[0:2] + full_days[:4],
        'val_dates': [weekends[2]] + full_days[4:-2] + [partial_days[3]]
    },
    'second_split': {
        'train_dates': partial_days[1:4] + weekends[1:3] + full_days[2:-2],
        'val_dates': [weekends[0]] + full_days[:2] + [partial_days[0]]
    },
    'third_split': {
        'train_dates': [partial_days[0]] + partial_days[2:4] + [weekends[0]] + weekends[2:3] + full_days[:2] + full_days[4:-2],
        'val_dates': [weekends[1]] + full_days[2:4] + [partial_days[1]]
    },
    'fourth_split': {
        'train_dates': partial_days[0:2] + partial_days[3:4] + weekends[0:2] + full_days[:4] + full_days[6:-2],
        'val_dates': [weekends[2]] + full_days[4:6] + [partial_days[2]]
    },
    'fifth_split': {
        'train_dates': partial_days[0:3] + weekends[:3] + full_days[1:4] + full_days[5:-3],
        'val_dates': [full_days[0], full_days[4], full_days[-3]] + [partial_days[3]]
    },
}

# no validation dataset excluded and no partial dates for validation
cv_splits = {
    'first_split': {
        'train_dates': partial_days + weekends[:3] + full_days[:6],
        'val_dates': [weekends[3]] + full_days[6:]
    },
    'second_split': {
        'train_dates': partial_days + weekends[1:4] + full_days[2:],
        'val_dates': [weekends[0]] + full_days[:2]
    },
    'third_split': {
        'train_dates': partial_days + [weekends[0]] + weekends[2:] + full_days[:2] + full_days[4:],
        'val_dates': [weekends[1]] + full_days[2:4]
    },
    'fourth_split': {
        'train_dates': partial_days + weekends[0:2] + weekends[3:] + full_days[:4] + full_days[6:],
        'val_dates': [weekends[2]] + full_days[4:6]
    },
    'fifth_split': {
        'train_dates': partial_days + weekends + full_days[1:4] + full_days[5:-1],
        'val_dates': [full_days[0], full_days[4], full_days[-1]]
    },
}


def get_cv_days(splits: dict) -> list:
    cv_days = []
    for split_name, split_dict in splits.items():
        train_dates = split_dict['train_dates']
        test_dates = split_dict['val_dates']
        cv_days += train_dates + test_dates

    cv_days = list(set(cv_days))
    cv_days.sort()
    return cv_days


def get_index_for_days(data_index: pd.DatetimeIndex, list_of_days: list) -> np.ndarray:
    days = pd.to_datetime(list_of_days)
    mask = data_index.floor('D').isin(days)
    indices = np.where(mask)[0]
    return indices


def create_time_features(df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    """
    Create comprehensive time-based features for building occupancy prediction.

    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with time series index and sensor readings
    feature_columns : list or None
        List of feature columns to process. If None, all columns except 'Occupancy' will be used.

    Returns:
    --------
    pandas DataFrame
        Original DataFrame with additional engineered features
    """
    result_df = df.copy()

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != 'Occupancy']

    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Input DataFrame does not have a DatetimeIndex. Features may not be calculated correctly.")

    for col in feature_columns:
        result_df[f'{col}_lag_1min'] = df[col].shift(1)
        result_df[f'{col}_lag_5min'] = df[col].shift(5)
        result_df[f'{col}_lag_10min'] = df[col].shift(10)

        result_df[f'{col}_change'] = df[col].diff()

        result_df[f'{col}_rolling_mean_15min'] = df[col].rolling(window=15, min_periods=1).mean()
        result_df[f'{col}_rolling_mean_10min'] = df[col].rolling(window=10, min_periods=1).mean()
        result_df[f'{col}_rolling_mean_5min'] = df[col].rolling(window=5, min_periods=1).mean()

        result_df[f'{col}_rolling_std_15min'] = df[col].rolling(window=15, min_periods=1).std()
        result_df[f'{col}_rolling_std_10min'] = df[col].rolling(window=10, min_periods=1).std()
        result_df[f'{col}_rolling_std_5min'] = df[col].rolling(window=5, min_periods=1).std()

        changes = df[col].diff()

        result_df[f'{col}_rolling_change_mean_15min'] = changes.rolling(window=15, min_periods=1).mean()
        result_df[f'{col}_rolling_change_mean_10min'] = changes.rolling(window=10, min_periods=1).mean()
        result_df[f'{col}_rolling_change_mean_5min'] = changes.rolling(window=5, min_periods=1).mean()

        result_df[f'{col}_rolling_change_std_15min'] = changes.rolling(window=15, min_periods=1).std()
        result_df[f'{col}_rolling_change_std_10min'] = changes.rolling(window=10, min_periods=1).std()
        result_df[f'{col}_rolling_change_std_5min'] = changes.rolling(window=5, min_periods=1).std()

        result_df[f'{col}_pct_change'] = df[col].pct_change() * 100

        result_df[f'{col}_roc_5min'] = (df[col] - df[col].shift(5)) / 5

    result_df = result_df.bfill()

    return result_df


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=1, names=["Index", "Timestamp", "Temperature", "Humidity", "Light", "CO2",
                                                   "HumidityRatio", "Occupancy"])

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df.set_index("Timestamp", inplace=True)

    numeric_columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio", "Occupancy"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)

    df = create_time_features(df, feature_columns=features)

    return df


def get_data() -> tuple:
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    file_path = os.path.join(current_dir, '..', 'dataset', 'datatraining.txt')
    df_training = load_data(file_path)

    file_path = os.path.join(current_dir, '..', 'dataset', 'datatest.txt')
    df_test = load_data(file_path)

    file_path = os.path.join(current_dir, '..', 'dataset', 'datatest2.txt')
    df_test_2 = load_data(file_path)

    df_test['num'] = 1
    df_training['num'] = 2
    df_test_2['num'] = 3

    df = pd.concat([df_test, df_training, df_test_2])
    return df, df_test, df_training, df_test_2


def filter_dataset(df: pd.DataFrame, remove_weekends: bool = False,
                   start_time: str = None, end_time: str = None) -> pd.DataFrame:
    """
    Filter time series data based on weekdays and time ranges.

    Parameters:
    -----------
    df : pandas.DataFrame
        Time series DataFrame with datetime index
    remove_weekends : bool, default False
        Whether to remove weekends (Saturday and Sunday)
    start_time : str, default None
        Start time in 'HH:MM' format, e.g. '07:00'
    end_time : str, default None
        End time in 'HH:MM' format, e.g. '19:00'

    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame
    """
    filtered_df = df.copy()

    if remove_weekends:
        filtered_df = filtered_df[filtered_df.index.weekday < 5]

    if start_time is not None and end_time is not None:
        filtered_df = filtered_df.between_time(start_time, end_time)

    return filtered_df


def split_dataset_by_dates(
        df: pd.DataFrame,
        features: list,
        target: str,
        train_dates: list,
        test_dates: list,
        val_dates: list = None,
) -> tuple:
    """
    Split time series data into train, test, and validation sets based on date ranges.

    Parameters:
    -----------
    df : pandas.DataFrame
        Time series DataFrame with datetime index
    features : list
        List of feature column names
    target : str
        Target column name
    train_dates : list
        List of date strings in format 'YYYY-MM-DD' for training data
    test_dates : list
        List of date strings in format 'YYYY-MM-DD' for testing data
    val_dates : list, default None
        List of date strings in format 'YYYY-MM-DD' for validation data
        If None or empty list, validation sets will be empty

    Returns:
    --------
    tuple
        (X_train, y_train, X_test, y_test, X_val, y_val)
        Where X_* are numpy arrays of features and y_* are numpy arrays of targets
        If val_dates is None or empty, X_val and y_val will be empty arrays
    """
    train_dates = sorted(pd.to_datetime(train_dates))
    test_dates = sorted(pd.to_datetime(test_dates))

    train_mask = df.index.floor('D').isin(train_dates)
    test_mask = df.index.floor('D').isin(test_dates)

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]

    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, target]

    if val_dates and len(val_dates) > 0:
        val_dates = sorted(pd.to_datetime(val_dates))
        val_mask = df.index.floor('D').isin(val_dates)
        X_val = df.loc[val_mask, features]
        y_val = df.loc[val_mask, target]
    else:
        X_val = np.array([]).reshape(0, len(features))
        y_val = np.array([])

    return X_train, y_train, X_test, y_test, X_val, y_val


def evaluate_classification_model(model: Pipeline,
                                  X_train: pd.DataFrame,
                                  y_train: pd.DataFrame,
                                  X_test: pd.DataFrame,
                                  y_test: pd.DataFrame,
                                  X_val: pd.DataFrame = None,
                                  y_val: pd.DataFrame = None,
                                  feature_names: list = None
                                  ) -> dict:
    """
    Evaluate classification model performance on train, test, and validation sets.

    Parameters:
    -----------
    model : trained model or pipeline
        The trained classification model to evaluate
    X_train, X_test, X_val : pd.DataFrame
        Feature matrices for train, test, and validation sets
    y_train, y_test, y_val : pd.DataFrame
        Target vectors for train, test, and validation sets
    feature_names : list, default None
        Names of the features (columns)

    Returns:
    --------
    dict
        Dictionary containing evaluation metrics and predicted values
    """
    results = {}

    def get_metrics(X: pd.DataFrame, y: pd.DataFrame, set_name: str) -> dict:
        y_pred = model.predict(X)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X)[:, 1]

        metrics = {
            f"{set_name}_accuracy": accuracy_score(y, y_pred),
            f"{set_name}_precision": precision_score(y, y_pred),
            f"{set_name}_recall": recall_score(y, y_pred),
            f"{set_name}_f1": f1_score(y, y_pred),
            f"{set_name}_predictions": y_pred,
        }

        if y_proba is not None:
            metrics[f"{set_name}_probabilities"] = y_proba
            metrics[f"{set_name}_roc_auc"] = roc_auc_score(y, y_proba)

        cm = confusion_matrix(y, y_pred)
        metrics[f"{set_name}_confusion_matrix"] = cm

        cr = classification_report(y, y_pred, output_dict=True)
        metrics[f"{set_name}_classification_report"] = cr

        return metrics

    results.update(get_metrics(X_train, y_train, "train"))
    results.update(get_metrics(X_test, y_test, "test"))

    if X_val is not None and y_val is not None and len(X_val) > 0:
        results.update(get_metrics(X_val, y_val, "val"))

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        model_in_pipeline = model.named_steps["model"]

        if hasattr(model_in_pipeline, "feature_importances_"):
            importances = model_in_pipeline.feature_importances_
            indices = np.argsort(importances)[::-1]

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            results["feature_importances"] = {
                "names": [feature_names[i] for i in indices],
                "scores": [importances[i] for i in indices]
            }
        elif hasattr(model_in_pipeline, "coef_"):
            coefs = model_in_pipeline.coef_[0]
            indices = np.argsort(np.abs(coefs))[::-1]

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(coefs))]

            results["feature_importances"] = {
                "names": [feature_names[i] for i in indices],
                "scores": [coefs[i] for i in indices]
            }
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]

        results["feature_importances"] = {
            "names": [feature_names[i] for i in indices],
            "scores": [importances[i] for i in indices]
        }
    elif hasattr(model, "coef_"):
        coefs = model.coef_[0]
        indices = np.argsort(np.abs(coefs))[::-1]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coefs))]

        results["feature_importances"] = {
            "names": [feature_names[i] for i in indices],
            "scores": [coefs[i] for i in indices]
        }

    print("Model Evaluation Summary:")
    print("-" * 40)

    for set_name in ["train", "test", "val"]:
        if f"{set_name}_accuracy" in results:
            print(f"{set_name.capitalize()} Set Metrics:")
            print(f"  Accuracy:  {results[f'{set_name}_accuracy']:.4f}")
            print(f"  Precision: {results[f'{set_name}_precision']:.4f}")
            print(f"  Recall:    {results[f'{set_name}_recall']:.4f}")
            print(f"  F1 Score:  {results[f'{set_name}_f1']:.4f}")
            if f"{set_name}_roc_auc" in results:
                print(f"  ROC AUC:   {results[f'{set_name}_roc_auc']:.4f}")
            print("-" * 40)

    return results


def evaluate_and_visualize_model(model: Pipeline,
                                 X_train: pd.DataFrame,
                                 y_train: pd.DataFrame,
                                 X_test: pd.DataFrame,
                                 y_test: pd.DataFrame,
                                 X_val: pd.DataFrame = None,
                                 y_val: pd.DataFrame = None,
                                 train_idx: pd.DatetimeIndex = None,
                                 test_idx: pd.DatetimeIndex = None,
                                 val_idx: pd.DatetimeIndex = None,
                                 features_df: pd.DataFrame = None,
                                 features_to_plot: list = None):
    """
    Complete end-to-end evaluation and visualization of classification model.

    Parameters:
    -----------
    model : trained model or pipeline
        The trained classification model to evaluate
    X_train, X_test, X_val : pandas.DataFrame
        Feature matrices for train, test, and validation sets
    y_train, y_test, y_val : pandas.DataFrame
        Target vectors for train, test, and validation sets
    train_idx, test_idx, val_idx : pandas.DatetimeIndex
        Time indices for respective datasets
    features_df : pandas.DataFrame
        Original features DataFrame for plotting
    features_to_plot : str or list
        Feature(s) to plot alongside predictions

    Returns:
    --------
    tuple
        (evaluation_results, train_viz_df, test_viz_df, val_viz_df)
    """
    if hasattr(model, 'steps'):
        # exclude the model and scaler steps and log
        transform_steps = model.steps[1:-2]
        transform_pipeline = Pipeline(transform_steps)

        if isinstance(features_df, pd.DataFrame):
            features_df = transform_pipeline.transform(features_df)

    feature_names = features_df.columns
    print(f'input features: {feature_names}')
    eval_results = evaluate_classification_model(
        model, X_train, y_train, X_test, y_test,
        X_val, y_val, feature_names
    )

    results = {}

    if train_idx is not None:
        print("\n--- Training Set Misclassifications ---")
        train_viz_df = plot_misclassifications(
            train_idx, y_train, eval_results['train_predictions'],
            features_df, features_to_plot
        )
        results['train_viz_df'] = train_viz_df

    if test_idx is not None:
        print("\n--- Test Set Misclassifications ---")
        test_viz_df = plot_misclassifications(
            test_idx, y_test, eval_results['test_predictions'],
            features_df, features_to_plot
        )
        results['test_viz_df'] = test_viz_df

    if val_idx is not None and X_val is not None and len(X_val) > 0:
        print("\n--- Validation Set Misclassifications ---")
        val_viz_df = plot_misclassifications(
            val_idx, y_val, eval_results['val_predictions'],
            features_df, features_to_plot
        )
        results['val_viz_df'] = val_viz_df

    if 'feature_importances' in eval_results:
        plt.figure(figsize=(10, 6))
        importance_data = eval_results['feature_importances']
        y_pos = np.arange(len(importance_data['names']))

        plt.barh(y_pos, importance_data['scores'], align='center')
        plt.yticks(y_pos, importance_data['names'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()

    for set_name in ['train', 'test', 'val']:
        if f'{set_name}_confusion_matrix' in eval_results:
            plt.figure(figsize=(8, 6))
            cm = eval_results[f'{set_name}_confusion_matrix']

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Unoccupied', 'Occupied'],
                        yticklabels=['Unoccupied', 'Occupied'])

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {set_name.capitalize()} Set')
            plt.tight_layout()
            plt.show()

    return eval_results, results


def optimize_model_with_custom_splits(df: pd.DataFrame,
                                      features: list,
                                      target: str,
                                      cv_splitter: object,
                                      pipeline: Pipeline,
                                      param_distributions: dict,
                                      n_iter: int = 2,
                                      n_jobs: int = -1,
                                      scoring: str = 'f1',
                                      random_state=42) -> tuple:
    """
    Optimize model using RandomizedSearchCV with custom time series splits.
    """
    X = df[features]
    y = df[target]

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=0,
        refit=True,
        random_state=random_state,
        return_train_score=True,
    )

    search.fit(X, y)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")

    results = pd.DataFrame(search.cv_results_)

    split_scores = {}
    for i, (split_name, _) in enumerate(cv_splits.items()):
        split_scores[split_name] = {
            'test_score': results[f'split{i}_test_score'][search.best_index_],
            'train_score': results[f'split{i}_train_score'][search.best_index_]
        }

    print("\nPerformance across splits:")
    for split_name, scores in split_scores.items():
        print(f"  {split_name}: Test={scores['test_score']:.4f}, Train={scores['train_score']:.4f}")

    return search.best_estimator_, search.cv_results_, search.best_params_, search.best_score_
