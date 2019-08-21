import glob, tqdm, pickle
import numpy as np
import pandas as pd
import qml, gc
from dscribe.descriptors import ACSF
from ase.io import read

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

def expand_dataset(save_path=None):
    '''
    Add features and info to original train/test dataframes

    :param save_path: where to save expanded dataset if provided
    :return:
    '''

    # If more than 1 path possible for J coupling, pick p'th path
    p = 0

    # Load train/test datasets, as well as struct info
    train, test = pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv')

    # Load report dict
    with open(f'data/aux/rep_dict.pkl', 'rb') as h:
        rep_dict = pickle.load(h)

    for name, df in zip(['train', 'test'], [train, test]):
        expand_df = df.copy()#.iloc[:10000, :]

        # Num of bonds of the J interaction
        expand_df['chain'] = expand_df['type'].apply(lambda t: int(t[0]))

        accum = [] # Accumulate line computations in list to avoid writing to pandas df inside loop
        for i, row in tqdm.tqdm(enumerate(expand_df.itertuples(index=True)), total=expand_df.shape[0]):

            # Accumulator intermediate vars
            j3_middle_ixs = [np.nan, np.nan]
            j3_torsion_angle = np.nan
            j2_middle_ix = np.nan
            j2_bond_angle = np.nan

            bond_angles = rep_dict[row.molecule_name]['bond_angles']
            torsion_angles = rep_dict[row.molecule_name]['torsion_angles']

            # Fill middle index for 2J coups
            if row.chain == 2:
                valid_bonds = bond_angles[ # Consider both start-end or end-start
                    ((bond_angles[:, 0] == row.atom_index_0) & (bond_angles[:, -2] == row.atom_index_1)) |
                    ((bond_angles[:, 0] == row.atom_index_1) & (bond_angles[:, -2] == row.atom_index_0))
                , :]

                if valid_bonds.shape[0] > 1:
                    print(f'> Warning: 2J coupling from {row.atom_index_0:d} to {row.atom_index_1} on molecule'
                          f' {row.molecule_name} has {valid_bonds.shape[0]:d} possible paths with angles {valid_bonds[:,-1]}.'
                          f' Considering first path found.')

                j2_middle_ix = valid_bonds[p, 1]  # Added index of middle atom
                j2_bond_angle = valid_bonds[p, -1] # Bond angle features


            # Fill middle indexes for 3J coups
            elif row.chain == 3:
                valid_torsions = torsion_angles[ # Consider both start-end or end-start
                    ((torsion_angles[:, 0] == row.atom_index_0) & (torsion_angles[:, -2] == row.atom_index_1)) |
                    ((torsion_angles[:, 0] == row.atom_index_1) & (torsion_angles[:, -2] == row.atom_index_0))
                ,:]

                # If beginning of path in report is symmetric from specified path in train, pick correct order for middle 2 atoms
                rev_path = True if valid_torsions[p,0] == row.atom_index_1 else False

                if valid_torsions.shape[0] > 1:
                    print(f'> Warning: 3J coupling from {row.atom_index_0:d} to {row.atom_index_1} on molecule'
                          f' {row.molecule_name} has {valid_torsions.shape[0]:d} possible paths with angles {valid_torsions[:,-1]}.'
                          f' Considering first path found.')
                middle_ixs = valid_torsions[p, 1:3]
                if rev_path:
                    middle_ixs = middle_ixs[::-1]

                j3_middle_ixs = middle_ixs # Added indexes of middle atoms
                j3_torsion_angle = valid_torsions[p, -1] # Torsion angle features

            accum.append([j2_middle_ix, j3_middle_ixs[0], j3_middle_ixs[1], j2_bond_angle, j3_torsion_angle])

        extra_info = np.vstack(accum)
        # Assign accumulated info to columns in df
        col_names = [
            'j2_middle_ix',
            'j3_middle_ix_0',
            'j3_middle_ix_1',
            'j2_bond_angle',
            'j3_torsion_angle',
        ]
        for i, col_name in enumerate(col_names):
            expand_df[col_name] = extra_info[:, i]

        if save_path is not None:
            expand_df.to_hdf(save_path + '/' + f'{name}_expanded.h5', key='none', mode='w')

def angle_features(save_dir, prefix):
    train_extended, test_extended = pd.read_hdf('data/train_expanded.h5', mode='r'), pd.read_hdf('data/test_expanded.h5', mode='r')

    angles = ['j2_bond_angle', 'j3_torsion_angle']

    for name, df in zip(['train', 'test'], [train_extended, test_extended]):

        feats = np.vstack([np.array(df[col]) for col in angles] + [np.cos(df[col]) for col in angles]).T
        new_feats = pd.DataFrame(data=feats, columns=['j2_bond_angle', 'j3_torsion_angle', 'j2_bond_angle_cos', 'j3_torsion_angle_cos'])

        if save_dir is not None:
            new_feats.to_hdf(f'{save_dir}/{prefix}_{name}.h5', key='df', mode='w')

def distance_features(save_dir, prefix):
    train_extended, test_extended = pd.read_hdf('data/train_expanded.h5', mode='r'), pd.read_hdf(
        'data/test_expanded.h5', mode='r')
    struct = pd.read_csv('data/structures.csv')

    for name, df in zip(['train', 'test'], [train_extended, test_extended]):

        def map_atom_info(df, atom_ix):
            df = pd.merge(
                df, struct, how='left',
                left_on=['molecule_name', f'atom_index_{atom_ix:d}'],
                right_on=['molecule_name', f'atom_index'],
            )
            df = df.drop('atom_index', axis=1)
            df = df.rename(
                columns={
                    'atom' : f'atom_{atom_ix:d}',
                    'x' : f'x_{atom_ix}',
                    'y' : f'y_{atom_ix}',
                    'z' : f'z_{atom_ix}',
                }
            )
            return df

        df = map_atom_info(df, 0)
        df = map_atom_info(df, 1)

        df_p_0 = df[['x_0', 'y_0', 'z_0']].values
        df_p_1 = df[['x_1', 'y_1', 'z_1']].values

        dist = np.linalg.norm(df_p_0 - df_p_1, axis=1)

        dist = np.vstack(
            [
                dist,
                1/dist,
                1/dist**2,
                1/dist**3,
            ]
        ).T

        new_feats = pd.DataFrame(data=dist, columns=['dist', 'inv_dist', 'inv_dist_2', 'inv_dist_3'])

        if save_dir is not None:
            new_feats.to_hdf(f'{save_dir}/{prefix}_{name}.h5', key='df', mode='w')

def compute_acsf_descriptors(prefix, rcutoffs):

    species = ['H', 'C', 'N', 'O', 'F']
    g2_params = [
        [1, 0],
        # [1, 1],
        [1, 2],
        # [1, 3],
        # [1, 4],
        # [4, 1],
        [4, 2],
        # [4, 3],
        # [4, 4],
    ]
    g4_params = [
        [1, 1, 1],
        # [1, 4, 1],
        [1, 8, 1],
        # [1, 16, 1],
        # [1, 32, 1],
        # [1, 64, 1],
        [1, 1, -1],
        # [1, 4, -1],
        [1, 8, -1],
        # [1, 16, -1],
        # [1, 32, -1],
        # [1, 64, -1],
    ]
    # g5_params = [
    #     [1, 1, 1],
    #     # [1, 4, 1],
    #     [1, 8, 1],
    #     # [1, 16, 1],
    #     [1, 32, 1],
    #     # [1, 64, 1],
    #     [1, 1, -1],
    #     # [1, 4, -1],
    #     [1, 8, -1],
    #     # [1, 16, -1],
    #     [1, 32, -1],
    #     # [1, 64, -1],
    # ]
    featnames = ['g1'] +\
                [f'g2_{i:d}' for i in range(len(g2_params))] +\
                [f'g4_{i:d}' for i in range(len(g4_params) * 3)]# +\
                # [f'g5_{i:d}' for i in range(len(g5_params) * 3)]

    col_names = []
    for s in species:
        col_names.extend(
            [f'{s}_{fn}' for fn in featnames]
        )

    # Set up ACSF descriptor
    acsf = ACSF(
        g2_params=g2_params,
        g4_params=g4_params,
        # g5_params=g5_params,
        species=species,
        rcut=rcutoffs[0],
    )

    # Read mol info
    xyz_files = glob.glob('data/structures/*.xyz')
    mols = []
    for xyz_file in tqdm.tqdm(xyz_files, total=len(xyz_files)):
        mol = read(xyz_file, format='xyz')
        # print(mol.get_atomic_numbers())
        mols.append(mol)

    # Create ACSF output for all mols
    acsf_mol = acsf.create(mols, positions=None, n_jobs=4)

    # Save ACSF descriptors
    pd.DataFrame(
        data=acsf_mol, columns=col_names
    ).to_hdf(f'data/descriptors/{prefix}.h5', key='acsf', mode='w')

def compute_acsf_features(acsf_file, save_dir, prefix):

    struct = pd.read_csv('data/structures.csv')
    acsf_ = pd.read_hdf(f'data/descriptors/{acsf_file}.h5').astype(np.float16)
    acsf_cols = list(acsf_.columns)
    acsf = pd.concat([struct, acsf_], axis=1).astype({c : np.float16 for c in acsf_cols})
    del acsf_, struct
    gc.collect()

    def gen():
        yield 'train', pd.read_hdf('data/train_expanded.h5', mode='r')
        yield 'test', pd.read_hdf('data/test_expanded.h5', mode='r')

    for name, df in gen():

        def map_atom_info(df, atom_ix):
            df = pd.merge(
                df, acsf, how='left',
                left_on=['molecule_name', f'atom_index_{atom_ix:d}'],
                right_on=['molecule_name', f'atom_index'],
            )
            df = df.drop('atom_index', axis=1)

            new_names = {cname: f'a{atom_ix:d}_{cname}' for cname in acsf_cols}
            new_names['atom'] = f'atom_{atom_ix:d}'
            df = df.rename(
                columns=new_names
            )

            return df

        print('merge 0...')
        df = map_atom_info(df, 0)
        print('merge 1...')
        df = map_atom_info(df, 1)

        new_feat_cols = [f'a0_{cname}' for cname in acsf_cols] + [f'a1_{cname}' for cname in acsf_cols]
        new_feats = df.loc[:, new_feat_cols]

        if save_dir is not None:
            new_feats.to_hdf(f'{save_dir}/{prefix}_{name}.h5', key='df', mode='w')

if __name__ == '__main__':
    # expand_dataset('./data')
    # angle_features(save_dir='features', prefix='angle_feats')
    # distance_features(save_dir='features', prefix='simple_distance_v2')
    for r in [1.5, 3]:
        print(r)
        name=f'g2_g4_v2_0_r{r:.2f}'
        compute_acsf_descriptors(prefix=name, rcutoffs=[r])
        compute_acsf_features(acsf_file=name, save_dir='features', prefix=name)
    # calc_coloumb_matrices(size=29, save_path='data/descriptors/CMs_unsorted.pkl')
    # calc_CM_pair_features(max_terms=5, save_dir='features', save_name_prefix='cm_unsorted_maxterms_5')




