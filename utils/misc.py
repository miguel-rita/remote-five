import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_feature_importance(feature_importance_df, num_feats, relative_save_path=None):
    '''
    Save feature importance .png from computed fold feature importances

    :param feature_importance_df: pandas dataframe
        Dataframe containing feature importances per fold
    :param num_feats: int
        Maximum number of features to plot
    :param relative_save_path: string
        Relative path and name to store .png image with feature importances. If not provided, will show() image
    :return: --
    '''

    # =================================================================================================
    # Plot feat importance
    # =================================================================================================

    feats = feature_importance_df[['feature', 'importance']].groupby('feature').mean().sort_values('importance',
                                                                                                   ascending=False).reset_index().iloc[
            :num_feats, :2]  # Mean feat importance across folds
    y_size = np.min([feats.shape[0], num_feats])  # Num. of feats to plot
    y_pos = np.arange(feats.shape[0])

    fig, ax = plt.subplots(figsize=(14, y_size / 2.5))
    ax.barh(y_pos, feats.values[:, 1], align='center', color='#118715')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats.values[:, 0])
    ax.invert_yaxis()
    ax.set_title('LightGBM Feature importance (averaged over folds)')

    plt.tight_layout()
    if relative_save_path is not None:
        plt.savefig(relative_save_path)
    else:
        plt.show()
    plt.close()

def generate_sub(test_preds, sub_path):
    '''
    Generate submission file

    :param test_preds: np array
        Ordered 1d array of test predictions
    :param sub_name: str
        Path to sub, including name
    :return: --
    '''

    template = pd.read_csv('data/sample_submission.csv')
    template.loc[:, 'scalar_coupling_constant'] = test_preds
    template.to_csv(sub_path, index=False)
