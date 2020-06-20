import numpy as np
import scipy.sparse as sp
from keras.datasets import mnist as m
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph

def _grid(m1,m2, dtype=np.float32):
    """Returns the embedding of a grid graph."""
    M = m1*m2
    x = np.linspace(0, 1, m1, dtype=dtype)
    y = np.linspace(0, 1, m2, dtype=dtype)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2), dtype)
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z


def replace_random_edges(A, noise_level):
    """Replace randomly chosen edges by random edges."""
    M, M = A.shape
    n = int(noise_level * A.nnz // 2)

    indices = np.random.permutation(A.nnz//2)[:n]
    rows = np.random.randint(0, M, n)
    cols = np.random.randint(0, M, n)
    vals = np.random.uniform(0, 1, n)
    assert len(indices) == len(rows) == len(cols) == len(vals)

    A_coo = sp.triu(A, format='coo')
    assert A_coo.nnz == A.nnz // 2
    assert A_coo.nnz >= n
    A = A.tolil()

    for idx, row, col, val in zip(indices, rows, cols, vals):
        old_row = A_coo.row[idx]
        old_col = A_coo.col[idx]

        A[old_row, old_col] = 0
        A[old_col, old_row] = 0
        A[row, col] = 1
        A[col, row] = 1

    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def get_adj_from_data(X_l, Y_l=None, X_u=None, adj='knn', k=10, knn_mode='distance', metric='euclidean',
                      self_conn=True):

    if adj not in {'rbf', 'knn'}:
        raise ValueError('adj must be either rbf or knn')
    if X_u is not None:
        X = np.concatenate((X_l, X_u), axis=0)
    else:
        X = X_l

    # Compute transition prob matrix
    if adj == 'rbf':
        # Estimate bandwidth
        if Y_l is None:
            bw = 0.01
        else:
            bw = eval_bw(X_l, np.argmax(Y_l, axis=1))

        # Compute adjacency matrix
        d = squareform(pdist(X, metric='sqeuclidean'))
        A = np.exp(-d / bw)

        # No self-connections (avoids self-reinforcement)
        if self_conn is False:
            np.fill_diagonal(A, 0.0)
    elif adj == 'knn':
        if k is None:
            raise ValueError('k must be specified when adj=\'knn\'')
        # Compute adjacency matrix
        A = kneighbors_graph(
            X, n_neighbors=k,
            mode=knn_mode,
            metric=metric,
            include_self=self_conn
        ).toarray()
        A = sp.csr_matrix(np.maximum(A, A.T))
    else:
        raise NotImplementedError()

    return A


def grid_graph(m1,m2,k=5, corners=False):
    z = _grid(m1,m2)
    A = get_adj_from_data(z, adj='knn', k=k, metric='euclidean')

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = sp.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    return A

# test #
#A = grid_graph(1,5, k=3,corners=False)
#A = replace_random_edges(A, 0).astype(np.float32)
#print(A)
