import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder


def plot_misclassifications(df_index: pd.DatetimeIndex,
                            y_true: pd.DataFrame,
                            y_pred: pd.DataFrame,
                            features_df: pd.DataFrame = None,
                            feature_to_plot: list = None) -> pd.DataFrame:
    """
    Create plots showing when classification mistakes occurred.

    Parameters:
    -----------
    df_index : pandas.DatetimeIndex
        The original dataframe's datetime index
    y_true : numpy.ndarray
        True target values
    y_pred : numpy.ndarray
        Predicted target values
    features_df : pandas.DataFrame, default None
        DataFrame containing original features for context
    feature_to_plot : str or list, default None
        Feature(s) to plot alongside the predictions

    Returns:
    --------
    pandas.DataFrame
        DataFrame with true values, predictions, and error flags
    """
    results_df = pd.DataFrame(index=df_index)
    results_df['true'] = y_true
    results_df['pred'] = y_pred
    results_df['error'] = (y_true != y_pred).astype(int)
    results_df['false_negatives'] = results_df['error'] * results_df['true']  # Show only false negatives
    results_df['false_positives'] = results_df['error'] * (1 - results_df['true'])  # Show only false positives

    if features_df is not None:
        if feature_to_plot is not None:
            if isinstance(feature_to_plot, str):
                feature_to_plot = [feature_to_plot]
            for feature in feature_to_plot:
                if feature in features_df.columns:
                    results_df[feature] = features_df[feature]

    def assign_time_group(ts):
        hour = ts.hour
        if 0 <= hour < 7:
            return 'Night (00–07)'
        elif 7 <= hour < 19:
            return 'Day (07–19)'
        else:
            return 'Evening (19–24)'

    results_df['date'] = results_df.index.date
    results_df['time_group'] = results_df.index.map(assign_time_group)
    results_df['day_bucket'] = results_df['date'].astype(str) + ' / ' + results_df['time_group']

    daily_groups = list(results_df.groupby('day_bucket'))
    error_days = [date for date, group in daily_groups if group['error'].sum() > 0]
    print(f"Found classification errors on {len(error_days)} days.")

    for date, group in daily_groups:
        if group['error'].sum() > 0:
            weekday = group.index[0].strftime('%A')
            error_count = group['error'].sum()
            total_count = len(group)
            error_rate = error_count / total_count * 100

            print(
                f"Date: {date}, {weekday}: {error_count} errors out of {total_count} records ({error_rate:.2f}%)")

            plot_daily_results(group, feature_to_plot)

    return results_df


def plot_daily_results(df: pd.DataFrame,
                       features_to_plot: list = None) -> None:
    """
    Plot a day's worth of data with true values, predictions, and errors highlighted.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing at least 'true' and 'pred' columns
    features_to_plot : list, default None
        Additional feature(s) to plot on secondary axis
    """
    if features_to_plot is None:
        features_to_plot = [None]

    for feature in features_to_plot:

        fig, ax1 = plt.subplots(figsize=(14, 7))

        ax1.plot(df.index, df['true'], color='blue', alpha=0.3,
                 label='True Occupancy', marker='o', markersize=7)

        ax1.plot(df.index, df['pred'], color='green', alpha=0.7,
                 label='Predicted Occupancy', marker='x', markersize=4, linestyle='--')

        # Highlight false negatives (room is occupied but predicted as empty)
        false_neg = df[df['false_negatives'] == 1]
        if not false_neg.empty:
            ax1.scatter(false_neg.index, false_neg['true'], color='red', s=100,
                        marker='x', label='False Negative', zorder=5)

        # Highlight false positives (room is empty but predicted as occupied)
        false_pos = df[df['false_positives'] == 1]
        if not false_pos.empty:
            ax1.scatter(false_pos.index, false_pos['pred'], color='purple', s=100,
                        marker='+', label='False Positive', zorder=5)

        ax1.set_ylabel('Occupancy', color='black', fontsize=12)
        ax1.set_ylim(-0.1, 1.1)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Unoccupied', 'Occupied'])
        ax1.tick_params(axis='y', labelcolor='black')

        if feature is not None:
            ax2 = ax1.twinx()

            if feature in df.columns:
                ax2.plot(df.index, df[feature], color='orange', alpha=0.3,
                         label=feature, marker='.', markersize=3)

            ax2.set_ylabel(feature, color='orange', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='orange')

        ax1.set_xlabel('Timestamp', fontsize=12)
        ax1.legend(loc='center left')
        # title_date = df.index[0].strftime('%Y-%m-%d (%A)')
        # plt.title(f'Occupancy Predictions for {title_date}', fontsize=14)
        fig.tight_layout()
        # plt.savefig(f'figures/base_model/base_{title_date}_{feature}.jpg', bbox_inches='tight')
        plt.show()


def plot_param_importances(results: pd.DataFrame) -> None:
    """
    Plot parameter importances based on correlation with CV score.
    """
    param_names = [p for p in results.columns if p.startswith('param_')]

    importances = []
    for param in param_names:
        clean_name = param.replace('param_', '')

        param_values = results[param]

        if param_values.dtype == 'object':
            try:
                numeric_values = pd.to_numeric(param_values, errors='coerce')
                if numeric_values.notna().sum() > 0.5 * len(numeric_values):  # Mostly numeric
                    processed_values = numeric_values.fillna(numeric_values.mean())
                else:

                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    processed_values = encoder.fit_transform(param_values.values.reshape(-1, 1)).flatten()
            except Exception as e:
                print(f"Error processing parameter '{clean_name}': {e}")
                continue
        else:
            processed_values = param_values

        if len(np.unique(processed_values)) > 1:  # Skip constant parameters
            correlation = np.abs(np.corrcoef(processed_values, results['mean_test_score'])[0, 1])
            importances.append((clean_name, correlation))

    importances.sort(key=lambda x: x[1], reverse=True)

    if len(importances) > 0:
        plt.figure(figsize=(10, 6))
        names = [i[0] for i in importances]
        values = [i[1] for i in importances]
        plt.barh(names, values)
        plt.xlabel('Correlation with CV score')
        plt.title('Parameter Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid parameters found for importance calculation.")
