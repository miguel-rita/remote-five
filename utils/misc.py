import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os, pickle, tqdm

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

    fig, ax = plt.subplots(figsize=(14, y_size/2.5))
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

def expand_dataset(path_to_csv_dataset):
    '''
    Add index and position columns for all atoms along a J bond

    :param path_to_csv_dataset:
    :return:
    '''

    print('TODO')

def parse_obabel_reports(reports_dir, save_path=None):
    '''
    Parse obabel reports in dir into a dict.
    Dict keys will be molecule codes.
    Value is a dict with the following keys:
        'bond_angles': 2d np array, for each triplet (line) :[atom_number_0, atom_number_1, atom_number_2, bond_angle]
        'torsion_angles': 2d np array, for each quadruplet (line) :[atom_number_0, atom_number_1, atom_number_2, atom_number_3, torsion_angle]
    :param reports_dir:
    :param save_path: str
        If specified pickle dict as 'save_path'
    :return:
    '''

    rep_dict = {}
    rep_dirs = glob.glob(reports_dir + '/*.report')
    for rd in tqdm.tqdm(rep_dirs, total=len(rep_dirs)):
        mol_name = rd.split('/')[-1].split('.')[0]
        with open(rd, mode='r') as rep:

            lines = rep.readlines()
            first_bond_ix = lines.index('BOND ANGLES\n')
            first_torsion_ix = lines.index('TORSION ANGLES\n')
            last_bond_ix = first_torsion_ix - 3

            # Some reports have chirality info
            if 'CHIRAL ATOMS\n' not in lines:
                last_torsion_ix = len(lines) - 3
            else:
                last_torsion_ix = lines.index('CHIRAL ATOMS\n') - 3

            bond_array = np.array(
                    # First +1 to ignore header and second +1 to include last angle
                    [bond_line.split() for bond_line in lines[first_bond_ix+1:last_bond_ix+1]]
                )[:, [0,1,2,-1]].astype(float) # Pick only atom indexes and angle
            bond_array[:, :3] -= 1 # Remove one since obabel adds 1 to atom indexes (1-based)

            torsion_array = None
            # If molecule has torsion info
            if last_torsion_ix > first_torsion_ix:
                torsion_array = np.array(
                        [torsion_line.split() for torsion_line in lines[first_torsion_ix+1:last_torsion_ix+1]]
                    )[:, [0,1,2,3,-1]].astype(float)
                torsion_array[:, :4] -= 1 # Remove one since obabel adds 1 to atom indexes (1-based)

            rep_dict[mol_name] = {
                'bond_angles' : bond_array,
                'torsion_angles' : torsion_array,
            }

    if save_path is not None:
        with open(save_path, 'wb') as h:
            pickle.dump(rep_dict, h)

    return rep_dict

def main():
    parse_obabel_reports(
        reports_dir='../data/structures_report',
        save_path='../data/aux/rep_dict.pkl'
    )

if __name__ == '__main__':
    main()
