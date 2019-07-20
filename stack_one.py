import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae
from utils.preprocess import prepare_datasets
from utils.misc import generate_sub
from lgbm import LgbmModel

'''
Setup
'''

# Define custom metrics
def bmae(pred, true, types):
    log_maes = []
    for utype in np.unique(types):
        mask = types == utype
        utype_mae = np.max([mae(true[mask], pred[mask]), 1e-9])
        log_maes.append(np.log(utype_mae))
    return np.mean(log_maes)

def feval_func(preds, train_data):
    return (
        'bmae',
        bmae(
            pred=preds,
            true=train_data.get_label(),
            types=train_data.data.loc[:, 'type'].values,
        ),
        False  # True if higher values are better, False otherwise. Check LGBM docs.
    )

# Load dataset
feature_sets = [
    'cm_unsorted_maxterms_15',
    'angle_feats',
    'simple_distance',
]

x_train, x_test, y_tgt = prepare_datasets(
    feature_sets=feature_sets,
    augment_dataset=False,
)

# Select features
blacklist = ['id', 'molecule_name', 'atom_index_0', 'atom_index_1',]# 'j3_torsion_angle_cos', 'j2_bond_angle_cos',
             # 'sorted_CM_7_atom_0', 'sorted_CM_7_atom_1', 'sorted_CM_8_atom_0', 'sorted_CM_8_atom_1']
x_train.drop([c for c in x_train.columns if c in blacklist], inplace=True, axis=1)
x_test.drop([c for c in x_test.columns if c in blacklist], inplace=True, axis=1)

'''
Model training
'''

lgbm_model = LgbmModel()

fit = bool(0)
sub = bool(1)
model_name = 'v200k'

if fit:
    lgbm_model.fit(
        train=x_train,
        y_tgt=y_tgt,
        params={
            'objective': 'regression_l1',
            # 'metric': 'None',
            'boosting': 'gbdt',
            'num_leaves': 30,
            'min_data_in_leaf': 100,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 5,
            'learning_rate': 0.1,
            'verbose': 1,
        },
        run_params={
            'num_boost_round':200000,
            'early_stopping_rounds':100,
            'verbose_eval':100,
        },
        model_name=model_name,
        save_model=True,
        metric=bmae,
        feval=feval_func,
        save_feat_importances=True,
        random_seed=42
    )
else:
    model_dir = 'models/v200k_-0.90_2019-07-17 06:36:19'
    ts = model_dir.split('/')[-1]
    metric_val = re.search('_.?\d\.\d\d_', model_dir).group()
    lgbm_model.load(model_dir)
    y_preds = lgbm_model.predict(
        dataset=x_test,
        is_train=False,
    )
    generate_sub(y_preds, sub_path=f'subs/{model_name}{metric_val}{ts}.csv')
    # generate_sub(y_preds, sub_path=f'subs/{model_name}_debug.csv')
