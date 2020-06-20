import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras import backend as K

modes = {
    'S': 1,    # Single (rank(A)=2, rank(B)=2)
    'M': 2,    # Mixed (rank(A)=2, rank(B)=3)
    'iM': 3,   # Inverted mixed (rank(A)=3, rank(B)=2)
    'B': 4,    # Batch (rank(A)=3, rank(B)=3)
    'UNK': -1  # Unknown
}


################################################################################
# Ops for convolutions / Laplacians
def filter_dot(fltr, features):
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        if len(K.int_shape(fltr)) == 3:
            # Batch mode
            return K.batch_dot(fltr, features)
        else:
            # Mixed mode
            return mixed_mode_dot(fltr, features)


def normalize_A(A):
    D = degrees(A)
    D = tf.sqrt(D)[:, None] + K.epsilon()
    if K.ndim(A) == 3:
        # Batch mode
        output = (A / D) / transpose(D, perm=(0, 2, 1))
    else:
        # Single mode
        output = (A / D) / transpose(D)

    return output


def degrees(A):
    if K.is_sparse(A):
        D = tf.sparse.reduce_sum(A, axis=-1)
    else:
        D = tf.reduce_sum(A, axis=-1)

    return D


def degree_matrix(A, return_sparse_batch=False):
    D = degrees(A)

    batch_mode = K.ndim(D) == 2
    N = tf.shape(D)[-1]
    batch_size = tf.shape(D)[0] if batch_mode else 1

    inner_index = tf.tile(tf.stack([tf.range(N)] * 2, axis=1), (batch_size, 1))
    if batch_mode:
        if return_sparse_batch:
            outer_index = repeat(
                tf.range(batch_size), tf.ones(batch_size) * tf.cast(N, tf.float32)
            )
            indices = tf.concat([outer_index[:, None], inner_index], 1)
            dense_shape = (batch_size, N, N)
        else:
            return tf.linalg.diag(D)
    else:
        indices = inner_index
        dense_shape = (N, N)

    indices = tf.cast(indices, tf.int64)
    values = tf.reshape(D, (-1, ))
    return tf.SparseTensor(indices, values, dense_shape)


################################################################################
# Scipy to tf.sparse conversion
def sp_matrix_to_sp_tensor_value(x):
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return tf.SparseTensorValue(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def sp_matrix_to_sp_tensor(x):
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


################################################################################
# Matrix multiplication
def matmul_A_B(A, B):
    mode = autodetect_mode(A, B)
    if mode == modes['S']:
        # Single mode
        output = single_mode_dot(A, B)
    elif mode == modes['M']:
        # Mixed mode
        output = mixed_mode_dot(A, B)
    elif mode == modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = K.dot(A, B)
    elif mode == modes['B']:
        # Batch mode
        # Works only with dense tensors
        output = K.batch_dot(A, B)
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


def matmul_AT_B_A(A, B):
    mode = autodetect_mode(A, B)
    if mode == modes['S']:
        # Single (rank(A)=2, rank(B)=2)
        output = single_mode_dot(single_mode_dot(transpose(A), B), A)
    elif mode == modes['M']:
        # Mixed (rank(A)=2, rank(B)=3)
        output = mixed_mode_dot(transpose(A), B)
        if K.is_sparse(A):
            output = transpose(
                mixed_mode_dot(transpose(A), transpose(output, (0, 2, 1))),
                (0, 2, 1)
            )
        else:
            output = K.dot(output, A)
    elif mode == modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = mixed_mode_dot(B, A)
        output = K.batch_dot(transpose(A, (0, 2, 1)), output)
    elif mode == modes['B']:
        # Batch (rank(A)=3, rank(B)=3)
        # Works only with dense tensors
        output = K.batch_dot(
            K.batch_dot(
                transpose(A, (0, 2, 1)),
                B
            ),
            A
        )
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


def matmul_AT_B(A, B):
    mode = autodetect_mode(A, B)
    if mode == modes['S']:
        # Single (rank(A)=2, rank(B)=2)
        output = single_mode_dot(transpose(A), B)
    elif mode == modes['M']:
        # Mixed (rank(A)=2, rank(B)=3)
        output = mixed_mode_dot(transpose(A), B)
    elif mode == modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = K.dot(transpose(A, (0, 2, 1)), B)
    elif mode == modes['B']:
        # Batch (rank(A)=3, rank(B)=3)
        # Works only with dense tensors
        output = K.batch_dot(transpose(A, (0, 2, 1)), B)
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


def matmul_A_BT(A, B):
    mode = autodetect_mode(A, B)
    if mode == modes['S']:
        # Single (rank(A)=2, rank(B)=2)
        output = single_mode_dot(A, transpose(B))
    elif mode == modes['M']:
        # Mixed (rank(A)=2, rank(B)=3)
        output = mixed_mode_dot(A, transpose(B, (0, 2, 1)))
    elif mode == modes['iM']:
        # Inverted mixed (rank(A)=3, rank(B)=2)
        # Works only with dense tensors
        output = K.dot(A, transpose(B))
    elif mode == modes['B']:
        # Batch (rank(A)=3, rank(B)=3)
        # Works only with dense tensors
        output = K.batch_dot(A, transpose(B, (0, 2, 1)))
    else:
        raise ValueError('A and B must have rank 2 or 3.')

    return output


################################################################################
# Ops related to the modes of operation (single, mixed, batch)
def autodetect_mode(A, X):
    if K.ndim(X) == 2:
        if K.ndim(A) == 2:
            return modes['S']
        elif K.ndim(A) == 3:
            return modes['iM']
        else:
            return modes['UNK']
    elif K.ndim(X) == 3:
        if K.ndim(A) == 2:
            return modes['M']
        elif K.ndim(A) == 3:
            return modes['B']
        else:
            return modes['UNK']
    else:
        return modes['UNK']


def single_mode_dot(A, B):
    a_sparse = K.is_sparse(A)
    b_sparse = K.is_sparse(B)
    if a_sparse and b_sparse:
        raise ValueError('Sparse x Sparse matmul is not implemented yet.')
    elif a_sparse:
        output = tf.sparse_tensor_dense_matmul(A, B)
    elif b_sparse:
        output = transpose(
            tf.sparse_tensor_dense_matmul(
                transpose(B), transpose(A)
            )
        )
    else:
        output = tf.matmul(A, B)

    return output


def mixed_mode_dot(A, B):
    s_0_, s_1_, s_2_ = K.int_shape(B)
    B_T = transpose(B, (1, 2, 0))
    B_T = reshape(B_T, (s_1_, -1))
    output = single_mode_dot(A, B_T)
    output = reshape(output, (s_1_, s_2_, -1))
    output = transpose(output, (2, 0, 1))

    return output


################################################################################
# Wrappers for automatic switching between dense and sparse ops
def transpose(A, perm=None, name=None):
    if K.is_sparse(A):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(A, perm=perm, name=name)


def reshape(A, shape=None, name=None):
    if K.is_sparse(A):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(A, shape=shape, name=name)


################################################################################
# Misc ops
def matrix_power(x, k):
    if K.ndim(x) != 2:
        raise ValueError('x must have rank 2.')
    sparse = K.is_sparse(x)
    if sparse:
        x_dense = tf.sparse.to_dense(x)
    else:
        x_dense = x

    x_k = x_dense
    for _ in range(k - 1):
        x_k = K.dot(x_k, x_dense)

    if sparse:
        return tf.contrib.layers.dense_to_sparse(x_k)
    else:
        return x_k


def repeat(x, repeats):
    x = tf.expand_dims(x, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    arr_tiled = tf.tile(x, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def segment_top_k(x, I, ratio, top_k_var):
    num_nodes = tf.segment_sum(tf.ones_like(I), I)  # Number of nodes in each graph
    cumsum = tf.cumsum(num_nodes)  # Cumulative number of nodes (A, A+B, A+B+C)
    cumsum_start = cumsum - num_nodes  # Start index of each graph
    n_graphs = tf.shape(num_nodes)[0]  # Number of graphs in batch
    max_n_nodes = tf.reduce_max(num_nodes)  # Order of biggest graph in batch
    batch_n_nodes = tf.shape(I)[0]  # Number of overall nodes in batch
    to_keep = tf.ceil(ratio * tf.cast(num_nodes, tf.float32))
    to_keep = tf.cast(to_keep, tf.int32)  # Nodes to keep in each graph

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(cumsum_start, I)) + (I * max_n_nodes)

    y_min = tf.reduce_min(x)
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    # subtract 1 to ensure that filler values do not get picked
    dense_y = dense_y * tf.cast(y_min - 1, tf.float32)
    # top_k_var is a variable with unknown shape defined in the elsewhere
    dense_y = tf.assign(top_k_var, dense_y, validate_shape=False)
    dense_y = tf.scatter_update(dense_y, index, x)
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    perm = tf.argsort(dense_y, direction='DESCENDING')
    perm = perm + cumsum_start[:, None]
    perm = tf.reshape(perm, (-1,))

    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    mask = repeat(to_rep, rep_times)

    perm = tf.boolean_mask(perm, mask)

    return perm
