import re, os, time, datetime, pickle, glob
import numpy as np
import pandas as pd
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold
from utils.misc import save_feature_importance

class LgbmModel:

    def __init__(self):
        self.bsts = [] # Boosters for each fold
        self.fold_info = [] # List of tuples of form (fold_num, train_ixs, val_ixs)

    def fit(self, train, y_tgt, params, run_params, nfolds, nfolds_to_run, metric, model_name, save_model, save_feat_importances, random_seed, feval=None):

        '''
        Fit lgbm booster using CV

        :param train:
        :param y_tgt:
        :param params:
        :param run_params:
        :param nfolds: int
            Number of CV folds
        :param nfolds_to_run:
            Number of CV folds to compute
        :param metric:
        :param model_name:
        :param save_model:
        :param save_feat_importances:
        :param random_seed:
        :param feval:
        :return:
        '''

        if nfolds_to_run is None:
            nfolds_to_run = nfolds

        # Setup CV folds
        fold_metrics = []
        feature_importance_df = pd.DataFrame()
        y_oof = np.zeros(y_tgt.size, dtype=np.float32)

        for fold_num, (train_ix, val_ix) in enumerate(
            StratifiedKFold(
                n_splits=nfolds,
                random_state=random_seed,
                shuffle=True,
            ).split(X=train, y=train.loc[:,'type'].values)
        ):
            self.fold_info.append((fold_num, train_ix, val_ix))

        if save_model:
            timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            full_model_name = f'{model_name}_{timestamp}'
            print('> lgbm : Creating save dir . . .')
            model_dir = f'models/{full_model_name}'
            if not os.path.exists(model_dir):
                os.makedirs(f'models/{full_model_name}')

        # Train one booster per fold
        for fold_num, train_ix, val_ix in self.fold_info:

            if fold_num + 1 > nfolds_to_run:
                print(f'> lgbm: Done training only {nfolds_to_run} fold(s).')
                break

            print(f'> lgbm : Training on fold number {fold_num} . . .')

            x_train_fold = train.iloc[train_ix, :]
            x_val_fold = train.iloc[val_ix, :]
    
            train_data = lgbm.Dataset(data=x_train_fold, label=y_tgt[train_ix], free_raw_data=True)
            val_data = lgbm.Dataset(data=x_val_fold, label=y_tgt[val_ix], free_raw_data=False)

            bst = lgbm.train(
                params,
                train_data,
                valid_sets = val_data,
                # feval = feval,
                num_boost_round = run_params['num_boost_round'],
                early_stopping_rounds = run_params['early_stopping_rounds'],
                verbose_eval = run_params['verbose_eval'],
            )

            self.bsts.append(bst)

            oof_pred = bst.predict(x_val_fold)

            y_oof[val_ix] = oof_pred
            meanoof = np.mean(oof_pred)

            # OOF metric
            oof_metric = metric(
                pred=oof_pred,
                true=y_tgt[val_ix],
                types=x_val_fold.loc[:, 'type'].values,
            )
            fold_metrics.append(oof_metric)
            print(f'> lgbm : Fold metric : {oof_metric:.4f}')

            if save_model:
                print('> lgbm : Saving fold booster . . .')

                # Save fold info
                with open(f'{model_dir}/fold_info.pkl', 'wb') as h:
                    pickle.dump(self.fold_info, h)

                bst.save_model(f'{model_dir}/bst_{fold_num:d}_metric_{oof_metric:.4f}.bst')

            # Feature importances
            if save_feat_importances:
                fold_importance_df = pd.DataFrame()
                fold_importance_df["feature"] = train.columns
                fold_importance_df["importance"] = bst.feature_importance(importance_type='split')
                fold_importance_df["fold"] = fold_num + 1
                feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        print('> lgbm : CV results :')
        print(pd.Series(fold_metrics).describe())

        if save_feat_importances: # After saving boosters, to prevent unknown interaction between mpl and lgbm
            save_feature_importance(
                feature_importance_df=feature_importance_df,
                num_feats=100,
                relative_save_path=f'./importances/{full_model_name}.png',
            )

    def load(self, model_dir):
        if not os.path.exists(model_dir):
            raise NotADirectoryError(f'Cannot load from {model_dir}. Invalid directory.')

        # Load fold info
        with open(f'{model_dir}/fold_info.pkl', 'rb') as fold_info:
            self.fold_info = pickle.load(fold_info)

        # Load boosters in fold order
        bst_paths = glob.glob(f'{model_dir}/bst*')
        self.bsts = [None]*len(bst_paths)
        for bst_path in bst_paths:
            bst_num = int(re.search('bst_\d+_', bst_path).group()[4:-1])
            print(f'Loading bst number {bst_num:d} at path {bst_path}')
            self.bsts[bst_num] = lgbm.Booster(model_file=bst_path)
        print('Done loading')

    def predict(self, dataset, is_train):
        '''
        Predicts using boosters from all folds. If predict sub, may use less than all available folds

        :param dataset: pandas df
            Train or test dataset
        :param is_train: bool
            If true will compute oof predictions. If false will compute average across folds
        :return: np array
            Predictions
        '''

        if not self.bsts:
            raise AssertionError('No trained lgbm boosters found. Fit or load boosters before predicting.')
        else:
            print(f'> lgbm : Found {len(self.bsts):d} boosters.')

        y_oof = np.zeros(dataset.shape[0])
        y_test_preds = np.zeros((dataset.shape[0], len(self.bsts)))

        for (fold_num, train_ixs, val_ixs), bst in zip(self.fold_info, self.bsts):
            print(f'> lgbm : Predicting fold number {fold_num} . . .')
            if is_train:
                x_val_fold = dataset.iloc[val_ixs, :]
                oof_pred = bst.predict(x_val_fold)
                y_oof[val_ixs] = oof_pred

            else: # Predict test values
                y_test_preds[:, fold_num] = bst.predict(dataset)

        if not is_train:
            return np.mean(y_test_preds, axis=1)
        else:
            return y_oof
