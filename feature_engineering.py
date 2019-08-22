import numpy as np
import pandas as pd
import tqdm, gc
from numba import jit
from openbabel import *
from pybel import readfile
from utils.misc import get_gps_feature_cols, find_line
from utils.numba_utils import *

'''
Func definitions
'''

def gen_lookups(df, gb_col):
    '''
    return a cumsum array starting at 0 and a dict to map gb_col to its positions
    '''
    ss = df.groupby(gb_col).size().cumsum()
    ssx = np.zeros(len(ss)+1).astype(int)
    ssx[1:] = ss.values
    ssdict = {}
    for i, col in enumerate(ss.index):
        ssdict[col] = i
    return ssx, ssdict

@jit(nopython=True)
def numba_dist_matrix(mol_xyz):
    '''
    Credit @CPMP
    '''
    # return locs
    num_atoms = mol_xyz.shape[0]
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            d = np.sqrt((mol_xyz[i, 0] - mol_xyz[j, 0]) ** 2 + (mol_xyz[i, 1] - mol_xyz[j, 1]) ** 2 + (
                    mol_xyz[i, 2] - mol_xyz[j, 2]) ** 2)
            dmat[i, j] = d
            dmat[j, i] = d
    return dmat

@jit(nopython=True)
def numba_gps_features(xyz, pairs, atoms, MAX_N_FEAT_ATOMS, filter=-1):
    '''
    :param xyz:
    :param pairs:
    :param atoms:
    :param MAX_N_FEAT_ATOMS:
    :param filter: if -1 consider all atom types
    :return:
    '''
    N_PAIRS = pairs.shape[0]  # Number of scalar coupling pairs for this molecule
    N_CORE_ATOMS = 4  # Number of core atoms to serve as representation basis
    N_ATOMS = xyz.shape[0]  # Number of atoms in molecule
    dist_mat = numba_dist_matrix(xyz)

    gps_feats = np.empty((N_PAIRS, (N_CORE_ATOMS + 1) * MAX_N_FEAT_ATOMS)).astype(np.float32)
    gps_feats.fill(np.nan)  # Placeholder
    for i in range(N_PAIRS):

        # Determine distance to pair center for all molecule atoms
        a0 = pairs[i, 0]
        a1 = pairs[i, 1]
        center = (xyz[a0, :] + xyz[a1, :]) * 0.5

        mask = np.ones(N_ATOMS)
        if filter != -1:
            for a in range(N_ATOMS):
                if atoms[a] != filter:
                    mask[a] = 0
            mask[a0] = 1
            mask[a1] = 1
        N_MASK_ATOMS = int(np.sum(mask))
        masked_ixs = np.empty(N_MASK_ATOMS).astype(np.int16)
        mix = 0
        for i_ in range(N_ATOMS):
            if mask[i_] == 1:
                masked_ixs[mix] = i_
                mix += 1

        square_ds_to_center = np.empty(N_MASK_ATOMS).astype(np.float32)
        square_ds_to_center.fill(np.nan)

        for j in range(N_MASK_ATOMS):
            mj = masked_ixs[j]
            square_ds_to_center[j] = (xyz[mj, 0] - center[0]) ** 2.0 + (xyz[mj, 1] - center[1]) ** 2.0 + (
                    xyz[mj, 2] - center[2]) ** 2.0

        # Get argsort
        asort_ds_to_center = np.argsort(square_ds_to_center)
        asort_after_mask = np.copy(asort_ds_to_center)

        # If masked get correct masked ixs
        for j in range(N_MASK_ATOMS):
            pos_in_orig = asort_ds_to_center[j]
            asort_after_mask[j] = masked_ixs[pos_in_orig]

        # Get core atom indexes
        feat_atoms = [a0, a1]
        for aix in asort_ds_to_center:
            if aix != a0 and aix != a1:
                feat_atoms.append(aix)
            if len(feat_atoms) == min(MAX_N_FEAT_ATOMS, N_MASK_ATOMS):
                break
        core_atoms = feat_atoms[:min(N_CORE_ATOMS, N_MASK_ATOMS)]
        # At this stage for 3 atom molecules core_atoms has len() 3, but no problem since our feat array is initialized with np.nan by default

        for s in range(min(MAX_N_FEAT_ATOMS, N_MASK_ATOMS)):
            feat_atom_ix = feat_atoms[s]
            for t in range(min(N_CORE_ATOMS, N_MASK_ATOMS)):
                core_atom_ix = core_atoms[t]
                gps_feats[i, s * (N_CORE_ATOMS + 1) + t] = dist_mat[feat_atom_ix, core_atom_ix]
            # Add feat_atom type
            gps_feats[i, s * (N_CORE_ATOMS + 1) + t + 1] = atoms[feat_atom_ix]

    return gps_feats

@jit(nopython=True)
def numba_cyl_features(xyz, pairs, rs):
    N_PAIRS = pairs.shape[0]  # Number of scalar coupling pairs for this molecule
    N_ATOMS = xyz.shape[0]  # Number of atoms in molecule
    N_RS = rs.shape[0] # Number of radius thresholds

    dist_mat = numba_dist_matrix(xyz)

    out = np.zeros((N_PAIRS, N_RS))
    for j in range(N_RS):

        r = rs[j]

        for i in range(N_PAIRS):
            a = pairs[i, 0]
            b = pairs[i, 1]
            ab = xyz[b,:] - xyz[a,:]
            nab = dist_mat[a,b]

            ct = 0 # counter for atoms inside cylinder
            for k in range(N_ATOMS):

                if k==a or k==b:
                    continue

                ak = xyz[k,:] - xyz[a,:]
                bk = xyz[k,:] - xyz[b,:]
                nak = dist_mat[a, k]
                nbk = dist_mat[b, k]

                # Check if k atom not behind atoms
                cosa = ab.dot(ak) / (nab * nak)
                cosb = bk.dot(-ab) / (nab * nbk)

                if (cosa >= 0) & (cosb >= 0): # k atom is valid. lets check ortho dist to cylinder axis
                    d = np.sqrt(nak**2 * (1 - cosa**2))
                    if d <= r:
                        ct += 1

            out[i, j] = ct

    return out

def test_cyl():
    xyz = np.array([
        [0,0,0],
        [0,1,0],
        [0,1,1],
        [0,1,2],
        [0,2,0],
        [0,3,1],
        [0,-1,1],
    ]).astype(float)
    pairs = np.array([
        [0, 4],
        [1, 3],
        [2, 5],
    ])
    rs = np.array([0.5, 1.5, 5])
    print(numba_cyl_features(xyz, pairs, rs))

@jit(nopython=True)
def numba_geminal_ixs(triplet, angle_mat):

    empty_ixs = np.empty(0).astype(np.int16)
    a = triplet[0]
    b = triplet[1]
    c = triplet[2]

    # Check if invalid triplet (not applicable for this coupling type)
    # Also get col argixs
    col_ixs = np.ones(3)*-1
    n_valid_ixs = 0
    for i in range(3):
        ix = triplet[i]
        if ix == -2:
            return empty_ixs
        if ix != -1:
            col_ixs[i] = i
            n_valid_ixs += 1

    valid_ixs = np.zeros(n_valid_ixs)
    j = 0
    for i in range(3):
        if col_ixs[i] != -1:
            valid_ixs[j] = col_ixs[i]
            j += 1

    # Normal ixs search
    angle_submat = numba_extract_concat_cols(valid_ixs.astype(np.int16), angle_mat)
    ixs1, revs1 = numba_find_line(triplet[triplet != -1], angle_submat)

    # Mask output for symmetric queries
    if not ((a != -1) & (b == -1) & (c != -1)):
        ixs1 = ixs1[revs1 != 1]

    # Reversed ixs search - imagine we have H=9 X=0 Y=1, with angle info in format 1 0 9
    # If we looking at triplet 9 -1 -1 we won't find it, and we should
    # Note that generally 9 -1 -1 is a different feature from -1 -1 9 (in the latter we'd have Y=9, not H=9
    # In the case of a 2JHH bond, it is indeed the same feature (redundant)
    valid_ixs = 2 - valid_ixs[::-1] # geminal angles, 3 elements means max ix 2 (reverse to restore since subtraction also reverses)
    angle_submat = numba_extract_concat_cols(valid_ixs.astype(np.int16), angle_mat)
    ixs2, revs2 = numba_find_line(triplet[triplet != -1][::-1], angle_submat) # also reverse query triplet

    # Mask output for symmetric queries
    if not ((a != -1) & (b == -1) & (c != -1)):
        ixs2 = ixs2[revs2 != 1]

    # Union of both queries
    ixs = numba_union_arr(ixs1, ixs2)

    return ixs

def numba_vicinal_ixs(quadruplet, torsion_mat):

    empty_ixs = np.empty(0).astype(np.int16)
    a = quadruplet[0]
    b = quadruplet[1]
    c = quadruplet[2]
    d = quadruplet[3]

    # Check if invalid quadruplet (not applicable for this coupling type)
    # Also get col argixs
    col_ixs = np.ones(4)*-1
    n_valid_ixs = 0
    for i in range(4):
        ix = quadruplet[i]
        if ix == -2:
            return empty_ixs
        if ix != -1:
            col_ixs[i] = i
            n_valid_ixs += 1

    valid_ixs = np.zeros(n_valid_ixs)
    j = 0
    for i in range(4):
        if col_ixs[i] != -1:
            valid_ixs[j] = col_ixs[i]
            j += 1

    # Normal ixs search
    torsion_submat = numba_extract_concat_cols(valid_ixs.astype(np.int16), torsion_mat)
    ixs1, revs1 = numba_find_line(quadruplet[quadruplet != -1], torsion_submat)

    # Mask output for symmetric queries
    if (b != c) | (a != d):
        ixs1 = ixs1[revs1 != 1]

    # Reversed ixs search - imagine we have H=9 X=0 Y=1, with angle info in format 1 0 9
    # If we looking at triplet 9 -1 -1 we won't find it, and we should
    # Note that generally 9 -1 -1 is a different feature from -1 -1 9 (in the latter we'd have Y=9, not H=9
    # In the case of a 2JHH bond, it is indeed the same feature (redundant)
    valid_ixs = 3 - valid_ixs[::-1] # geminal angles, 3 elements means max ix 2 (reverse to restore since subtraction also reverses)
    torsion_submat = numba_extract_concat_cols(valid_ixs.astype(np.int16), torsion_mat)
    ixs2, revs2 = numba_find_line(quadruplet[quadruplet != -1][::-1], torsion_submat) # also reverse query triplet

    # Mask output for symmetric queries
    if (b != c) | (a != d):
        ixs2 = ixs2[revs2 != 1]

    # Union of both queries
    ixs = numba_union_arr(ixs1, ixs2)

    return ixs

# @jit(nopython=True)
def numba_angle_features(mol_ptypes, angle_mat, angles, torsion_mat, torsions, nfeats):

    NPAIRS = mol_ptypes.shape[0]

    feats = np.empty(shape=(NPAIRS, nfeats), dtype=np.float32)
    feats.fill(np.nan)

    for i in range(NPAIRS):
        a = mol_ptypes[i, 0]
        b = mol_ptypes[i, 1]
        t = mol_ptypes[i, 2]

        # Initialize H X Y Z

        angle_mat_ends = numba_extract_concat_cols(np.array([0, 2], dtype=np.int16), angle_mat)
        if torsion_mat.size > 0: # if torsions exist
            torsion_mat_ends = numba_extract_concat_cols(np.array([0, 3], dtype=np.int16), torsion_mat)

        UNDEFINED_IX = -2 # -1 reserved for empty placeholder on feature variations below
        H = a
        Y = UNDEFINED_IX
        Z = UNDEFINED_IX
        if t == 0 or t == 1:
            X = b
        elif t == 2 or t == 3 or t == 4:
            x_ix, revs = numba_find_line(np.array([a, b]), angle_mat_ends)
            X = angle_mat[x_ix[0], 1] # Index 0 since H forms only 1 bond
            Y = b
        else:
            x_ixs, revs = numba_find_line(np.array([a, b]), torsion_mat_ends)
            # There is only one X for the same reason as in 2J, H can only form 1 bond
            # Note pairs are supplied always H first yet torsions might be computed
            # in both directions
            if revs[0]: # H found last, hence X is before last (pos 2)
                X_ix = 2
            else: # H found first, no reversion, hence X is just after first (pos 1)
                X_ix = 1
            X = torsion_mat[x_ixs[0], X_ix]

            # If 3JHH, Y can also be defined unambiguously
            if t==5:
                if revs[0]:  # H found last, hence Y after first (pos 1)
                    Y_ix = 1
                else:  # H found first, no reversion, hence Y is before last (pos 2)
                    Y_ix = 2
                Y = torsion_mat[x_ixs[0], Y_ix]

            Z = b

        # Get all possible angle/torsion partial/complete combinations
        angle_combs, torsion_combs = numba_get_angle_torsion_combinations(H, X, Y, Z)

        # Compute ixs and aggregations per combination

        AGGS = 4

        # Angles
        NANG_COMBS = angle_combs.shape[0]
        for j in range(NANG_COMBS):
            ixs = numba_geminal_ixs(angle_combs[j, :], angle_mat=angle_mat)
            NANGLES = ixs.size
            if NANGLES:
                agg_angles = np.cos(angles[ixs])
                feats[i, j*AGGS] = agg_angles.size
                feats[i, j*AGGS+1] = np.min(agg_angles)
                feats[i, j*AGGS+2] = np.max(agg_angles)
                feats[i, j*AGGS+3] = np.mean(agg_angles)

        COL_OFFSET = NANG_COMBS * AGGS

        # Torsions
        if torsion_mat.size > 0: # if torsions exist
            NTORS_COMBS = torsion_combs.shape[0]
            for j in range(NTORS_COMBS):
                ixs = numba_vicinal_ixs(torsion_combs[j, :], torsion_mat=torsion_mat)
                NTORSIONS = ixs.size
                if NTORSIONS:
                    agg_torsions = torsions[ixs]
                    feats[i, COL_OFFSET + j * AGGS] = agg_torsions.size
                    feats[i, COL_OFFSET + j * AGGS + 1] = np.min(agg_torsions)
                    feats[i, COL_OFFSET + j * AGGS + 2] = np.max(agg_torsions)
                    feats[i, COL_OFFSET + j * AGGS + 3] = np.mean(agg_torsions)

    return feats

@jit(nopython=True)
def numba_experimental_features(xyz, pairs):
    N_ATOMS = xyz.shape[0]  # Number of atoms in molecule
    N_PAIRS = pairs.shape[0]  # Number of scalar coupling pairs for this molecule

    # dist_mat = numba_dist_matrix(xyz)

    exp_feats = np.empty((N_PAIRS, 1)).astype(np.float32)
    exp_feats.fill(np.nan)  # Placeholder

    exp_feats[:, :] = N_ATOMS

    return exp_feats

'''
Feature engineering
'''

def gps_standard_features(max_num_feat_atoms, save_dir, prefix):

    # Load data
    structures = pd.read_csv('data/structures.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Fast lookups
    xyz_ssx, xyz_dict = gen_lookups(structures, 'molecule_name')
    train_ssx, train_dict = gen_lookups(train, 'molecule_name')
    test_ssx, test_dict = gen_lookups(test, 'molecule_name')
    xyz_atom = structures['atom'].map({'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}).values.astype(np.uint8)

    # Numpy data
    xyz = structures[['x', 'y', 'z']].values.astype(np.float32)
    train_pairs = train[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    train_types = train[['type']].values
    test_pairs = test[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    test_types = test[['type']].values

    # For train and test
    for df_name, df, df_pairs, df_types, df_ssx, df_dict in zip(
            ['train', 'test'],
            [train, test],
            [train_pairs, test_pairs],
            [train_types, test_types],
            [train_ssx, test_ssx],
            [train_dict, test_dict],
    ):
        print(f'{df_name} feature engineering ...')

        # For each molecule
        feats = []
        unique_mols = df['molecule_name'].unique()
        for mol in tqdm.tqdm(unique_mols, total=unique_mols.shape[0]):

            mol_ix = xyz_dict[mol]
            mol_pairs_ix = df_dict[mol]
            mol_xyz = xyz[xyz_ssx[mol_ix]:xyz_ssx[mol_ix + 1], :]
            mol_pairs = df_pairs[df_ssx[mol_pairs_ix]:df_ssx[mol_pairs_ix + 1], :]
            mol_types = df_types[df_ssx[mol_pairs_ix]:df_ssx[mol_pairs_ix + 1], :]
            mol_atom = xyz_atom[xyz_ssx[mol_ix]:xyz_ssx[mol_ix + 1]]

            # Core calculations
            gps_feats = numba_gps_features(mol_xyz, mol_pairs, mol_atom, max_num_feat_atoms, filter=-1)
            gps_feats_h = numba_gps_features(mol_xyz, mol_pairs, mol_atom, max_num_feat_atoms, filter=1)
            gps_feats_c = numba_gps_features(mol_xyz, mol_pairs, mol_atom, max_num_feat_atoms, filter=6)
            gps_feats_n = numba_gps_features(mol_xyz, mol_pairs, mol_atom, max_num_feat_atoms, filter=7)
            gps_feats_o = numba_gps_features(mol_xyz, mol_pairs, mol_atom, max_num_feat_atoms, filter=8)

            feats.append(np.hstack([
                gps_feats,
                gps_feats_h,
                gps_feats_c,
                gps_feats_n,
                gps_feats_o,
            ]))

        # Define feature names and save to disk
        feat_names = get_gps_feature_cols(max_num_feat_atoms, include_redundant=True)
        feat_names += get_gps_feature_cols(max_num_feat_atoms, include_redundant=True, prefix='H')
        feat_names += get_gps_feature_cols(max_num_feat_atoms, include_redundant=True, prefix='C')
        feat_names += get_gps_feature_cols(max_num_feat_atoms, include_redundant=True, prefix='N')
        feat_names += get_gps_feature_cols(max_num_feat_atoms, include_redundant=True, prefix='O')
        pd.DataFrame(
            data=np.vstack(feats),
            columns=feat_names,
            dtype=np.float32,
        ).to_hdf(f'{save_dir}/{prefix}_{df_name}.h5', key='df', mode='w')

def cyl_standard_features(radii, save_dir, prefix):

    # Load data
    structures = pd.read_csv('data/structures.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Fast lookups
    xyz_ssx, xyz_dict = gen_lookups(structures, 'molecule_name')
    train_ssx, train_dict = gen_lookups(train, 'molecule_name')
    test_ssx, test_dict = gen_lookups(test, 'molecule_name')

    # Numpy data
    xyz = structures[['x', 'y', 'z']].values.astype(np.float32)
    train_pairs = train[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    train_types = train[['type']].values
    test_pairs = test[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    test_types = test[['type']].values

    # For train and test
    for df_name, df, df_pairs, df_types, df_ssx, df_dict in zip(
            ['train', 'test'],
            [train, test],
            [train_pairs, test_pairs],
            [train_types, test_types],
            [train_ssx, test_ssx],
            [train_dict, test_dict],
    ):
        print(f'{df_name} feature engineering ...')

        # For each molecule
        feats = []
        unique_mols = df['molecule_name'].unique()
        for mol in tqdm.tqdm(unique_mols, total=unique_mols.shape[0]):

            mol_ix = xyz_dict[mol]
            mol_pairs_ix = df_dict[mol]
            mol_xyz = xyz[xyz_ssx[mol_ix]:xyz_ssx[mol_ix + 1], :]
            mol_pairs = df_pairs[df_ssx[mol_pairs_ix]:df_ssx[mol_pairs_ix + 1], :]
            mol_types = df_types[df_ssx[mol_pairs_ix]:df_ssx[mol_pairs_ix + 1], :]

            # Core calculations
            cyl_feats = numba_cyl_features(mol_xyz, mol_pairs, np.array(radii))

            feats.append(np.hstack([
                cyl_feats,
            ]))

        # Define feature names and save to disk
        feat_names = [f'cyl_r_{r:.2f}' for r in radii]
        pd.DataFrame(
            data=np.vstack(feats),
            columns=feat_names,
            dtype=np.float32,
        ).to_hdf(f'{save_dir}/{prefix}_{df_name}.h5', key='df', mode='w')

def core_features(save_dir, prefix):

    TYPE_DICT = {
        '1JHN': 0,
        '1JHC': 1,
        '2JHH': 2,
        '2JHN': 3,
        '2JHC': 4,
        '3JHH': 5,
        '3JHC': 6,
        '3JHN': 7,
    }

    feature_names = []
    aggs = ['num', 'min', 'max', 'avg']
    for n in get_angle_torsion_agg_names():
        feature_names.extend([f'{n}_{agg}' for agg in aggs])

    # Load data
    structures = pd.read_csv('data/structures.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Fast lookups
    train_ssx, train_dict = gen_lookups(train, 'molecule_name')
    test_ssx, test_dict = gen_lookups(test, 'molecule_name')

    # Numpy data
    train_pairs = train[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    train_types = train['type'].map(TYPE_DICT).values.astype(np.uint8)
    train_ptypes = np.hstack([train_pairs, train_types[:, None]])
    train_uniques = train['molecule_name'].unique()
    test_pairs = test[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    test_types = test['type'].map(TYPE_DICT).values.astype(np.uint8)
    test_ptypes = np.hstack([test_pairs, test_types[:, None]])
    test_uniques = test['molecule_name'].unique()

    del train, test, structures
    gc.collect()

    # For train and test
    for df_name, df_uniques, df_ptypes, df_ssx, df_dict in zip(
            ['train', 'test'],
            [train_uniques, test_uniques],
            [train_ptypes, test_ptypes],
            [train_ssx, test_ssx],
            [train_dict, test_dict],
    ):
        print(f'{df_name} feature engineering ...')

        # For each molecule
        feature_chunks = []
        # for mol_name in tqdm.tqdm(['dsgdb9nsd_035567'], total=df_uniques.shape[0]):
        for mol_name in tqdm.tqdm(df_uniques, total=df_uniques.shape[0]):
            for mol in readfile('xyz', f'data/structures/{mol_name}.xyz'):
                mol = mol.OBMol

            # Bond types matrix to allow for custom features for each bond type
            mol_pairs_ix = df_dict[mol_name]
            mol_ptypes = df_ptypes[df_ssx[mol_pairs_ix]:df_ssx[mol_pairs_ix + 1], :]

            # Fill orders, angles and torsions matrices

            bonds = []
            for bond in OBMolBondIter(mol):
                beg_ix = bond.GetBeginAtomIdx() - 1
                end_ix = bond.GetEndAtomIdx() - 1
                order = bond.GetBondOrder()
                bonds.append([beg_ix, end_ix, order])
            bond_mat = np.vstack(bonds)

            angles = []
            for angle in OBMolAngleIter(mol):
                # WARNING : angles come with center atom first, hence the 1 0 2 order to compute the correct angle
                angle_in_degrees = mol.GetAngle(mol.GetAtom(angle[1] + 1), mol.GetAtom(angle[0] + 1),
                                                mol.GetAtom(angle[2] + 1))
                angles.append([angle[1], angle[0], angle[2], angle_in_degrees])
            angle_mat = np.vstack(angles)

            torsions = []
            for torsion in OBMolTorsionIter(mol):
                torsion_in_degrees = mol.GetTorsion(torsion[0] + 1, torsion[1] + 1, torsion[2] + 1, torsion[3] + 1)
                torsions.append([torsion[0], torsion[1], torsion[2], torsion[3], torsion_in_degrees])
            if torsions:
                torsion_mat = np.vstack(torsions)
                torsion_mat_ixs = torsion_mat[:, :4].astype(np.int16)
                torsion_mat_angles = torsion_mat[:, 4].astype(np.float32)
            else:
                torsion_mat_ixs = np.empty(0)
                torsion_mat_angles = np.empty(0)


            # Feat engineering
            feats = numba_angle_features(
                mol_ptypes=mol_ptypes.astype(np.int16),
                angle_mat=angle_mat[:,:3].astype(np.int16),
                angles=angle_mat[:,3].astype(np.float32),
                torsion_mat=torsion_mat_ixs.astype(np.int16),
                torsions=torsion_mat_angles.astype(np.float32),
                nfeats=len(feature_names),
            )
            feature_chunks.append(feats)


        pd.DataFrame(
            data=np.vstack(feature_chunks),
            columns=feature_names,
        ).astype(np.float16).to_hdf(f'{save_dir}/{prefix}_{df_name}.h5', key='df', mode='w')

def core_features_rings(save_dir, prefix):

    TYPE_DICT = {
        '1JHN': 0,
        '1JHC': 1,
        '2JHH': 2,
        '2JHN': 3,
        '2JHC': 4,
        '3JHH': 5,
        '3JHC': 6,
        '3JHN': 7,
    }

    # Load data
    structures = pd.read_csv('data/structures.csv')
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # Fast lookups
    train_ssx, train_dict = gen_lookups(train, 'molecule_name')
    test_ssx, test_dict = gen_lookups(test, 'molecule_name')

    # Numpy data
    train_pairs = train[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    train_types = train['type'].map(TYPE_DICT).values.astype(np.uint8)
    train_ptypes = np.hstack([train_pairs, train_types[:, None]])
    train_uniques = train['molecule_name'].unique()
    test_pairs = test[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
    test_types = test['type'].map(TYPE_DICT).values.astype(np.uint8)
    test_ptypes = np.hstack([test_pairs, test_types[:, None]])
    test_uniques = test['molecule_name'].unique()

    del train, test, structures
    gc.collect()

    ring_feature_names = [
        'path_n_rings',
        'max_path_ring_size',
        'max_path_num_rings'
    ]
    # ring_feature_names = []
    # for atom in ['x', 'y', 'z']:
    #     ring_feature_names.extend([f'{atom}_{n}' for n in ring_feature_base_names])

    # For train and test
    for df_name, df_uniques, df_ptypes, df_ssx, df_dict in zip(
            ['train', 'test'],
            [train_uniques, test_uniques],
            [train_ptypes, test_ptypes],
            [train_ssx, test_ssx],
            [train_dict, test_dict],
    ):
        print(f'{df_name} feature engineering ...')

        # For each molecule
        feature_chunks = []
        for mol_name in tqdm.tqdm(df_uniques, total=df_uniques.shape[0]):

            for mol in readfile('xyz', f'data/structures/{mol_name}.xyz'):
                mol = mol.OBMol

            # Bond types matrix to allow for custom features for each bond type
            mol_pairs_ix = df_dict[mol_name]
            mol_ptypes = df_ptypes[df_ssx[mol_pairs_ix]:df_ssx[mol_pairs_ix + 1], :]

            # Fill orders, angles and torsions matrices
            bonds = []
            for bond in OBMolBondIter(mol):
                beg_ix = bond.GetBeginAtomIdx()
                end_ix = bond.GetEndAtomIdx()
                order = bond.GetBondOrder()
                bonds.append([beg_ix - 1, end_ix - 1, order])
            bond_mat = np.vstack(bonds)

            angles = []
            for angle in OBMolAngleIter(mol):
                # WARNING : angles come with center atom first, hence the 1 0 2 order to compute the correct angle
                angle_in_degrees = mol.GetAngle(mol.GetAtom(angle[1] + 1), mol.GetAtom(angle[0] + 1),
                                                mol.GetAtom(angle[2] + 1))
                angles.append([angle[1], angle[0], angle[2], angle_in_degrees])
            angle_mat = np.vstack(angles)

            torsions = []
            for torsion in OBMolTorsionIter(mol):
                torsion_in_degrees = mol.GetTorsion(torsion[0] + 1, torsion[1] + 1, torsion[2] + 1, torsion[3] + 1)
                torsions.append([torsion[0], torsion[1], torsion[2], torsion[3], torsion_in_degrees])
            if torsions:
                torsion_mat = np.vstack(torsions)

            # Feat engineering

            NPAIRS = mol_ptypes.shape[0]
            fs = np.empty(shape=(NPAIRS, 12), dtype=np.float32)
            fs.fill(np.nan)

            # Ring info
            rmat = None
            ring_infos = []
            for ring in mol.GetSSSR():
                RSIZE = ring.Size()
                rixs = np.array(ring._path) - 1
                rsize = np.ones(RSIZE) * RSIZE
                ring_infos.append(np.vstack([rixs, rsize]))
            if ring_infos:
                rmat = np.hstack(ring_infos).T
                mn_mat = numba_fill_paths(abt=mol_ptypes, angle_mat=angle_mat, torsion_mat=torsion_mat)

                # Compute ring feats for x,y,z atoms
                NA = -1
                BLOCKSIZE = 4

                for k, ixs in enumerate([mol_ptypes[:, 1], mn_mat[:, 0], mn_mat[:, 1]]):
                    beg = k*BLOCKSIZE
                    end = beg + BLOCKSIZE
                    fs[:, beg:end] = numba_ring_mat_feats(rmat=rmat, ixs=ixs)

                nrings = np.sum(fs[:,[3,7,11]], axis=1, keepdims=True)
                maxrsize = np.max(fs[:,[1,5,9]], axis=1, keepdims=True)
                maxdelta = np.max(fs[:,[2,6,10]], axis=1, keepdims=True)
                fs = np.hstack([nrings, maxrsize, maxdelta])

                feature_chunks.append(fs)

        pd.DataFrame(
            data=np.vstack(feature_chunks),
            columns=ring_feature_names,
        ).astype(np.int16).to_hdf(f'{save_dir}/{prefix}_{df_name}.h5', key='df', mode='w')

if __name__ == '__main__':
    gps_standard_features(15, save_dir='features', prefix='gps_base_plus_h')
    # cyl_standard_features([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0], save_dir='features', prefix='cyl_feats')
    # core_features(save_dir='features', prefix='core_feats_angles_cos')
    # core_features_rings(save_dir='features', prefix='ring_feats_v2')
    # test_cyl()
