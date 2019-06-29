import glob, tqdm, pickle
import numpy as np
import pandas as pd
import qml

def inv_tri(cm_tri, size):
    '''
    Rebuild full matrix from concated upper triangle vector in
    fortran column-wise order

    :param cm_tri: flattened vector
    :param size: square matrix size
    :return:
    '''
    cm_tri = cm_tri[::-1]
    cm = np.zeros((size, size))
    for i in range(size):
        cm[:i + 1, i] = cm_tri[-(i + 1):][::-1]
        cm_tri = np.roll(cm_tri, i + 1)
    cm += np.triu(cm, k=1).T
    return cm

def calc_coloumb_matrices(size, save_path=None):
    '''
    Return dict of CMs for each molecule in both train and test sets

    :param size: int
        Max num of atoms per molecule found in datasets
    :param save_path: str
        If provided will pickle computed CMs to save_path
    :return: Dict of 2D numpy arrays
        Dict where keys are molecule_names, values are 2d CMs
    '''

    CMs = {}
    for file in tqdm.tqdm(glob.glob('./data/structures/*.xyz')):
        mol_name = file.split('/')[-1].split('.')[0]
        mol = qml.Compound(xyz=file)

        # After experiments, seems upper triangle CM was concated in fortran-order
        mol.generate_coulomb_matrix(size=size, sorting='unsorted')
        cm_tri = mol.representation
        cm = inv_tri(cm_tri, size=size)

        # Concat to dict
        CMs[mol_name] = cm

    if save_path is not None:
        with open(save_path, 'wb') as h:
            pickle.dump(CMs, h, protocol=pickle.HIGHEST_PROTOCOL)

    return CMs

def calc_CM_pair_features(max_terms, save_dir=None, save_name_prefix=None):
    '''
    First iteration of CM features

    :param max_terms: int
        Max CM interactions to keep per atom. Max 28 terms, since we have 29 atoms at most
    '''

    # Load data
    train, test = pd.read_csv('./data/train.csv'), pd.read_csv('./data/test.csv')
    with open('data/descriptors/CMs_unsorted.pkl', 'rb') as h:
        CMs = pickle.load(h)

    # Placeholders for new feature values
    new_feats = (np.zeros((train.shape[0], max_terms*2)), np.zeros((test.shape[0], max_terms*2)))

    for j, df in enumerate([train, test]):

        # For each entry
        for i,line in tqdm.tqdm(enumerate(df.values), total=df.shape[0]):
            mol_name, index_0, index_1 = str(line[1]), int(line[2]), int(line[3])
            cm = CMs[mol_name]

            # Grab the two relevant CM lines as feature
            line_0 = np.sort(cm[index_0, :])[::-1][:max_terms]
            line_1 = np.sort(cm[index_1, :])[::-1][:max_terms]
            new_feats[j][i, :] = np.hstack([line_0, line_1])

    feat_names = [f'sorted_CM_{n:d}_atom_0' for n in range(max_terms)] +\
                 [f'sorted_CM_{n:d}_atom_1' for n in range(max_terms)]

    if save_dir is not None:
        pd.DataFrame(data=new_feats[0], columns=feat_names).to_hdf(f'{save_dir}/{save_name_prefix}_train.h5', key='df', mode='w')
        pd.DataFrame(data=new_feats[1], columns=feat_names).to_hdf(f'{save_dir}/{save_name_prefix}_test.h5', key='df', mode='w')

if __name__ == '__main__':

    # calc_coloumb_matrices(size=29, save_path='data/descriptors/CMs_unsorted.pkl')
    calc_CM_pair_features(max_terms=15, save_dir='features', save_name_prefix='cm_unsorted_maxterms_15')




