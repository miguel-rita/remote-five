import numpy as np
import pandas as pd
import tqdm, glob, gc
from numba import jit
from tqdm import tqdm_notebook
from pybel import *
from openbabel import *
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sympy.geometry import Point3D

@jit(nopython=True)
def numba_find_line(line, arr, accept_reverse=True):
    N = arr.shape[0]
    a = line[0]
    b = line[1]
    NA = -1
    ixs = np.ones(N) * NA
    iii = 0
    for i in range(N):
        c = arr[i, 0]
        d = arr[i, 1]
        if ((a==c) & (b==d)) | ((a==d) & (b==c)):
            ixs[iii] = i
            iii += 1
    return ixs[ixs > NA].astype(np.uint16)

@jit(nopython=True)
def numba_args_where(num, vec):
    N = vec.shape[0]
    NA = -1
    ixs = np.ones(N) * NA
    iii = 0
    for i in range(N):
        c = vec[i]
        if c==num:
            ixs[iii] = i
            iii += 1
    return ixs[ixs > NA].astype(np.uint16)

@jit(nopython=True)
def numba_ring_mat_feats(rmat, ixs):
    '''
    Given rmat with 2 columns, where column 1 is atom_ix and column 2 is the size of the ring where it was found
    compute a number of ring features for atoms with ixs specified in 'ixs'

    One atom ix can appear more than once on rmat's column 1
    '''

    NFEATS = 4
    MAX_NUM_RINGS = 8
    NRMAT = rmat.shape[0]
    NIXS = ixs.shape[0]

    fmat = np.empty(shape=(NIXS, NFEATS), dtype=np.uint8)
    for i in range(NIXS):
        ix = ixs[i]
        rsf = np.zeros(MAX_NUM_RINGS, dtype=np.uint8)
        iii = 0
        for j in range(NRMAT):
            if rmat[j, 0] == ix:
                rsf[iii] = rmat[j, 1]
                iii += 1

        # default feat vals
        min_r_size = 0
        max_r_size = 0
        delta_r_size = 0
        nrings = iii

        # if rings found
        if nrings > 0:
            rsf = rsf[rsf > 0]
            min_r_size = np.min(rsf)
            max_r_size = np.max(rsf)
            delta_r_size = max_r_size - min_r_size

        fmat[i, 0] = min_r_size
        fmat[i, 1] = max_r_size
        fmat[i, 2] = delta_r_size
        fmat[i, 3] = nrings

    return fmat

@jit(nopython=True)
def numba_fill_paths(abt, angle_mat, torsion_mat):
    '''
    given abt, with pair ixs on first 2 cols and encoded bond type on 3rd col
    returns a 2col matrix with first col filled with middle ix for 2J or 1st and 2nd cols
    filled for 3J
    '''

    NA = -1
    mn = np.ones((abt.shape[0], 2)) * NA
    N_ABT = abt.shape[0]
    for i in range(N_ABT):
        pair = abt[i, :2]
        t = abt[i, 2]
        if t == 0 or t == 1:
            continue
        elif t == 2 or t == 3 or t == 4:
            sub_amat = np.vstack((angle_mat[:, 0], angle_mat[:, 2])).T
            angle_ixs = numba_find_line(pair, sub_amat)
            m_ix = angle_mat[angle_ixs[0], 1]
            mn[i, 0] = m_ix
        else:
            sub_tmat = np.vstack((torsion_mat[:, 0], torsion_mat[:, 3])).T
            torsion_ixs = numba_find_line(pair, sub_tmat)
            m_ix = torsion_mat[torsion_ixs[0], 1]
            n_ix = torsion_mat[torsion_ixs[0], 2]
            mn[i, 0] = m_ix
            mn[i, 1] = n_ix
    return mn

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

# # Load data
# structures = pd.read_csv('../data/structures.csv')
# train = pd.read_csv('../data/train.csv')
# test = pd.read_csv('../data/test.csv')
#
# # Fast lookups
# xyz_ssx, xyz_dict = gen_lookups(structures, 'molecule_name')
# train_ssx, train_dict = gen_lookups(train, 'molecule_name')
# test_ssx, test_dict = gen_lookups(test, 'molecule_name')
#
# TYPE_DICT = {
#     '1JHN':0,
#     '1JHC':1,
#     '2JHH':2,
#     '2JHN':3,
#     '2JHC':4,
#     '3JHH':5,
#     '3JHC':6,
#     '3JHN':7,
# }
#
# # Numpy data
# xyz = structures[['x', 'y', 'z']].values.astype(np.float32)
# xyz_atom = structures['atom'].map({'H':1, 'C':6, 'N':7, 'O':8, 'F':9}).values.astype(np.uint8)
# train_pairs = train[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
# train_types = train['type'].map(TYPE_DICT).values.astype(np.uint8)
# train_ptypes = np.hstack([train_pairs, train_types[:, None]])
# test_pairs = test[['atom_index_0', 'atom_index_1']].values.astype(np.int8)
# test_types = test['type'].map(TYPE_DICT).values.astype(np.uint8)
# test_ptypes = np.hstack([test_pairs, test_types[:, None]])
#
# mol_names = [g.split('/')[-1].split('.')[0] for g in glob.glob('../data/structures/*.xyz')[:]]
#
# ring_feature_base_names = [
#     'min_ring_size',
#     'max_ring_size',
#     'delta_ring_size',
#     'nrings',
# ]
# ring_feature_names = []
# for atom in ['x', 'y', 'z']:
#     ring_feature_names.extend([f'{atom}_{n}' for n in ring_feature_base_names])
#
# # 'dsgdb9nsd_005567'
# MOLNAME = 'dsgdb9nsd_035567'
# for mol_name in [MOLNAME]:
#     for mol in readfile('xyz', f'../data/structures/{mol_name}.xyz'):
#         mol = mol.OBMol
#
#     # Bond types matrix to allow for custom features for each bond type
#     mol_pairs_ix = train_dict[mol_name]
#     mol_ptypes = train_ptypes[train_ssx[mol_pairs_ix]:train_ssx[mol_pairs_ix + 1], :]
#
#     # Fill orders, angles and torsions matrices
#     bonds = []
#     for bond in OBMolBondIter(mol):
#         beg_ix = bond.GetBeginAtomIdx()
#         end_ix = bond.GetEndAtomIdx()
#         order = bond.GetBondOrder()
#         bonds.append([beg_ix - 1, end_ix - 1, order])
#     bond_mat = np.vstack(bonds)
#
#     angles = []
#     for angle in OBMolAngleIter(mol):
#         # WARNING : angles come with center atom first, hence the 1 0 2 order to compute the correct angle
#         angle_in_degrees = mol.GetAngle(mol.GetAtom(angle[1] + 1), mol.GetAtom(angle[0] + 1), mol.GetAtom(angle[2] + 1))
#         angles.append([angle[1], angle[0], angle[2], angle_in_degrees])
#     angle_mat = np.vstack(angles)
#
#     torsions = []
#     for torsion in OBMolTorsionIter(mol):
#         torsion_in_degrees = mol.GetTorsion(torsion[0] + 1, torsion[1] + 1, torsion[2] + 1, torsion[3] + 1)
#         torsions.append([torsion[0], torsion[1], torsion[2], torsion[3], torsion_in_degrees])
#     torsion_mat = np.vstack(torsions)
#
#     # Ring info
#
#     rmat = None
#     ring_infos = []
#     for ring in mol.GetSSSR():
#         RSIZE = ring.Size()
#         rixs = np.array(ring._path) - 1
#         rsize = np.ones(RSIZE) * RSIZE
#         ring_infos.append(np.vstack([rixs, rsize]))
#     if ring_infos:
#         rmat = np.hstack(ring_infos).T
#
#     # Feat engineering
#
#     NPAIRS = mol_ptypes.shape[0]
#     feature_chunk = np.empty(shape=(NPAIRS, len(ring_feature_base_names)), dtype=np.float32)
#     feature_chunk.fill(np.nan)
#     mn_mat = numba_fill_paths(abt=mol_ptypes, angle_mat=angle_mat, torsion_mat=torsion_mat)
#
#     # Compute ring feats for x,y,z atoms
#     rfeats = []
#     for ixs in [mol_ptypes[:,1], mn_mat[:,0], mn_mat[:,1]]:
#         rfeats.append(numba_ring_mat_feats(rmat=rmat, ixs=ixs))
#     rfeats = np.hstack(rfeats).astype(np.int16)
#
# rdf=pd.DataFrame(data=rfeats, columns=ring_feature_names)
# print(train.loc[train.molecule_name == MOLNAME, :].head(4))
# rfinal=pd.concat([train.loc[train.molecule_name == MOLNAME, ['atom_index_0', 'atom_index_1', 'type']].reset_index(), rdf.reset_index()], axis=1).head(20)
# print('done')
# a=pd.DataFrame(data=feature_chunk, columns=feature_names).astype(feature_names)
# pd.concat([train.loc[train.molecule_name == 'dsgdb9nsd_005567', :].reset_index(), a.reset_index()], axis=1).head(2)