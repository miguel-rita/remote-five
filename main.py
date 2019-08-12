import numpy as np
import pandas as pd
import time, datetime, gc, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from utils.misc import get_gps_feature_cols
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def train_predict_coupling_type(coupling_type, train, test, y_tgt, local_lgb_params,
                                sub, feat_cols, test_size=0.2, random_state=42):
    
    print(f'\n*** Training {coupling_type} model ***')

    y_data = y_tgt[train.type == coupling_type].astype('float32')
    X_data = train.loc[train.type == coupling_type, feat_cols].values.astype('float32')
    X_test = test.loc[test.type == coupling_type, feat_cols].values.astype('float32')

    print(f'X,y train shapes for {coupling_type} model:', X_data.shape, y_data.shape)

    # Train validation split
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=test_size, random_state=random_state)

    model = LGBMRegressor(**local_lgb_params)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
              verbose=100, early_stopping_rounds=100)

    y_val_pred = model.predict(X_val)
    val_score = np.log(mean_absolute_error(y_val, y_val_pred))
    print(f'{coupling_type} logMAE: {val_score}')

    y_pred = model.predict(X_test)

    sub.loc[test['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred

    return model, val_score

def select_feats_per_type(ctype):

    ring_feature_names = [
        'path_n_rings',
        'max_path_ring_size',
        'max_path_num_rings'
    ]
    cm_fn = [
        'sorted_CM_0_atom_0',
        'sorted_CM_1_atom_0',
        'sorted_CM_2_atom_0',
        'sorted_CM_3_atom_0',
        'sorted_CM_4_atom_0',
        'sorted_CM_5_atom_0',
        'sorted_CM_6_atom_0',
        'sorted_CM_7_atom_0',
        'sorted_CM_8_atom_0',
        'sorted_CM_9_atom_0',
        'sorted_CM_10_atom_0',
        'sorted_CM_11_atom_0',
        'sorted_CM_12_atom_0',
        'sorted_CM_13_atom_0',
        'sorted_CM_14_atom_0',
        'sorted_CM_0_atom_1',
        'sorted_CM_1_atom_1',
        'sorted_CM_2_atom_1',
        'sorted_CM_3_atom_1',
        'sorted_CM_4_atom_1',
        'sorted_CM_5_atom_1',
        'sorted_CM_6_atom_1',
        'sorted_CM_7_atom_1',
        'sorted_CM_8_atom_1',
        'sorted_CM_9_atom_1',
        'sorted_CM_10_atom_1',
        'sorted_CM_11_atom_1',
        'sorted_CM_12_atom_1',
        'sorted_CM_13_atom_1',
        'sorted_CM_14_atom_1',
    ]
    cm_a1_fn = [
        'sorted_CM_0_atom_1',
        'sorted_CM_1_atom_1',
        'sorted_CM_2_atom_1',
        'sorted_CM_3_atom_1',
        'sorted_CM_4_atom_1',
        'sorted_CM_5_atom_1',
        'sorted_CM_6_atom_1',
        'sorted_CM_7_atom_1',
        'sorted_CM_8_atom_1',
        'sorted_CM_9_atom_1',
        'sorted_CM_10_atom_1',
        'sorted_CM_11_atom_1',
        'sorted_CM_12_atom_1',
        'sorted_CM_13_atom_1',
        'sorted_CM_14_atom_1',
    ]
    radii_fm = [
        'cyl_r_0.50',
        'cyl_r_0.75',
        'cyl_r_1.00',
        'cyl_r_1.25',
        'cyl_r_1.50',
        'cyl_r_2.00',
        'cyl_r_3.00',
    ]
    gen_rfn = lambda c : [f'{c}_{n}' for n in ring_feature_names]

    # Commom features to all types
    feature_set = []#cm_fn
    feats_1J = cm_a1_fn#gen_rfn('x')
    feats_2J = cm_fn
    feats_3J = cm_fn

    # Type specific features
    if ctype == '1JHN':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(feats_1J)

    elif ctype == '1JHC':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(feats_1J)

    elif ctype == '2JHH':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(['angle_2J'])
        feature_set.extend(feats_2J)

    elif ctype == '2JHN':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(['angle_2J'])
        feature_set.extend(feats_2J)

    elif ctype == '2JHC':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(['angle_2J'])
        feature_set.extend(feats_2J)

    elif ctype == '3JHH':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(['torsion_3J_min'])
        feature_set.extend(feats_3J)

    elif ctype == '3JHC':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(['torsion_3J_min'])
        feature_set.extend(feats_3J)

    elif ctype == '3JHN':
        feature_set.extend(get_gps_feature_cols(n_atoms=10))
        feature_set.extend(['torsion_3J_min'])
        feature_set.extend(feats_3J)

    else:
        raise ValueError('Unknown coupling type')
    return feature_set

def load_datasets(featsets):
    
    print('Loading datasets ...')

    print('(Loading train)')
    train = pd.read_csv('data/train.csv')
    train_featsets = [pd.read_hdf(f'features/{fs}_train.h5') for fs in featsets]

    y_tgt = train.scalar_coupling_constant.values
    train.drop('scalar_coupling_constant', axis=1, inplace=True)

    final_train = pd.concat([train] + train_featsets, axis=1)
    del train, train_featsets
    gc.collect()

    print('(Loading test)')
    test = pd.read_csv('data/test.csv')
    test_featsets = [pd.read_hdf(f'features/{fs}_test.h5') for fs in featsets]

    final_test = pd.concat([test] + test_featsets, axis=1)
    del test, test_featsets
    gc.collect()
    
    print('Done loading datasets.')
    
    return final_train, final_test, y_tgt  

def stack_one():

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    logging.basicConfig(
        filename=f'logs/run_{timestamp}.log',
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S',
        level=logging.DEBUG,
    )

    featsets = [
        'gps_base',
        'cyl_feats',
        'core_feats',
        'cm_unsorted_maxterms_15'
    ]
    logging.info(f'Using feature sets: {featsets}')
    train, test, y_tgt = load_datasets(featsets=featsets)

    # Sub placeholder
    sub = pd.read_csv(f'data/sample_submission.csv')

    LGB_PARAMS = {
        'objective': 'regression',
        'metric': 'mae',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 256,
        'min_child_samples': 80,
        'n_estimators': 200,
        'n_jobs': -1,
    }
    logging.info(f'LGB parameters:')
    for k,v in LGB_PARAMS.items():
        logging.info(f'     {k}:{v}')

    val_log_maes = {}
    test_size=0.2
    logging.info(f'CV params:')
    logging.info(f'     test_size: {test_size:.2f}')

    # Setup feature importance directory
    imp_subdir = f'importances/run_{timestamp}'
    os.makedirs(imp_subdir)

    for ctype in train.type.unique():

        ctype_feats = select_feats_per_type(ctype)
        logging.info(f'{ctype} features used:')
        for f in ctype_feats:
            logging.info(f'     {f}')
        model, val_log_mae = train_predict_coupling_type(
            coupling_type=ctype,
            train=train,
            test=test,
            y_tgt=y_tgt,
            local_lgb_params=LGB_PARAMS,
            sub=sub,
            feat_cols=ctype_feats,
            test_size=test_size,
            random_state=42
        )
        val_log_maes[ctype] = val_log_mae

        # Feature importances
        df_importance = pd.DataFrame({'feature': ctype_feats, 'importance': model.feature_importances_})
        f, a = plt.subplots(1, 1, figsize=(10, 0.12 * len(ctype_feats)))
        imp_plot = sns.barplot(x="importance", y="feature", data=df_importance.sort_values('importance', ascending=False), ax=a)
        f.savefig(f'{imp_subdir}/{ctype}_{val_log_mae}.png')


    mean_loss = np.mean(list(val_log_maes.values()))
    final_log_mae = f'Final logMAE : {mean_loss:.4f}\n'
    logging.info(final_log_mae)
    print('\n', final_log_mae)
    for k,v in val_log_maes.items():
        logging.info(f'     {k}:{v}')
        print(k, v)

    # Save sub
    sub_name = '_'.join(featsets) + f'_{timestamp}' + f'_logMAE_{mean_loss:.4f}'
    sub.set_index('id').to_csv(f'subs/{sub_name}', float_format='%.3f')

if __name__ == '__main__':
    stack_one()