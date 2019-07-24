import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def prepare_datasets(feature_sets, augment_dataset, subsample_frac=None):
    '''
    Prepare train, test datasets as well as target vector

    :param feature_sets: list of str
        List of datasets to load excluding the _train or _test suffix
    :param augment_dataset: bool
        Swap atom pair indexes to duplicate features
    :param subsample_frac: float
        Train on a fraction of train data
    :return: pandas df (train), pandas df (test), np 1D array (target)
    '''

    # Load base features & info
    train, test = pd.read_csv('./data/train.csv'), pd.read_csv('./data/test.csv')
    train_original_cols = list(train.columns)

    # Concat train and test extra features
    train = pd.concat([train] + [pd.read_hdf(f'features/{fs}_train.h5') for fs in feature_sets], axis=1)
    test = pd.concat([test] + [pd.read_hdf(f'features/{fs}_test.h5') for fs in feature_sets], axis=1)

    # Encode categorical features
    le = LabelEncoder()
    le.fit(train['type'].unique())
    train['type'] = le.transform(train['type'])
    test['type'] = le.transform(test['type'])

    # Optional data augmentation
    if augment_dataset:
        max_terms = 5
        atom0_feats = [f'sorted_CM_{n:d}_atom_0' for n in range(max_terms)]
        atom1_feats = [f'sorted_CM_{n:d}_atom_1' for n in range(max_terms)]
        train_swapped = train[train_original_cols + atom1_feats + atom0_feats]
        train = pd.concat([train, train_swapped], axis=0, ignore_index=True, sort=False)

    # Isolate target
    target = train.loc[:, 'scalar_coupling_constant'].values
    train.drop(['scalar_coupling_constant'], axis=1, inplace=True)

    if subsample_frac is not None:
        for _, subsample_ixs in StratifiedKFold(
            n_splits=int(1/subsample_frac),
            random_state=42,
            shuffle=True,
        ).split(X=train, y=train.loc[:, 'type'].values):
            train = train.iloc[subsample_ixs, :]
            target = target[subsample_ixs]
            break

    return train, test, target
