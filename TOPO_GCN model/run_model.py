import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import keras
from keras import optimizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, \
    Activation, concatenate, Reshape, AveragePooling1D, MaxPooling1D, Conv1D, ZeroPadding2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from utils import _grid, replace_random_edges, get_adj_from_data, grid_graph
from ops import sp_matrix_to_sp_tensor, transpose, mixed_mode_dot, filter_dot
import scipy.sparse as sp
from geometric_learning import Exponentialtransformation, GraphConv

sensor_images = np.load('sensor_images.npy')
PI_null_images = np.load('PI_null_images.npy')
vec_PI_null_images = PI_null_images.reshape((-1,400,1))
PI_one_images = np.load('PI_one_images.npy')
vec_PI_one_images = PI_one_images.reshape((-1,400,1))
Longest_NON_Rotation_PDs_null = np.load('Longest_Rotation_PDs_null_collection.npy',allow_pickle=True)
labels = np.load('labels.npy')

train_images, test_images, train_PI_null, test_PI_null, train_PDs_null, test_PDs_null, train_labels, test_labels = train_test_split(
    sensor_images, vec_PI_null_images, Longest_NON_Rotation_PDs_null, labels, test_size=0.1, random_state=1)
#################################################

num_classes = 2
batch_size = 128
epochs = 10000
learning_rate = 1e-3
l2_reg = 5e-4
adj = grid_graph(20,20, k=5,corners=False)
adj = replace_random_edges(adj, 0).astype(np.float32)
N = 400

# Fractional-G-SSL part #
gamma = 0.1
degrees = np.array(adj.sum(1)).flatten()
degrees[np.isinf(degrees)] = 0.
D = sp.diags(degrees, 0)
L = D - adj

L_darray = L.toarray()
D, V = np.linalg.eigh(L_darray, 'U')
M_gamma_Lambda = D
M_gamma_Lambda[M_gamma_Lambda < 1e-10] = 0
M_V = V

M_gamma_Lambda = np.float_power(M_gamma_Lambda, gamma)
M_gamma_Lambda = np.diag(M_gamma_Lambda, 0)
M_gamma_Lambda = sp.csr_matrix(M_gamma_Lambda)
M_V = sp.csr_matrix(M_V)
Lg = M_V * M_gamma_Lambda
Lg = Lg * sp.csr_matrix.transpose(M_V)

Lg = Lg.toarray()
Lg = Lg.reshape(1, -1)
Lg[abs(Lg) < 1e-10] = 0.
Lg = Lg.reshape(N, -1)
Dg = np.diag(np.diag(Lg))
Ag = Dg - Lg
Ag = sp.csr_matrix(Ag, dtype=np.float32)

alpha = 0.1
power_Dg = np.float_power(np.diag(Dg), -alpha)
power_Dg = np.diag(power_Dg)
power_Dg = sp.csr_matrix(power_Dg, dtype=np.float32)

power_Dg_right = np.float_power(np.diag(Dg), (alpha - 1))
power_Dg_right = np.diag(power_Dg_right)
power_Dg_right = sp.csr_matrix(power_Dg_right, dtype=np.float32)

fltr = power_Dg * Ag
fltr = fltr * power_Dg_right

A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr))
#################################################

# regular convolution #
def initial_sensor_convolution_layers(input_img):
    model = ZeroPadding2D((1, 1), input_shape=(9, 20, 1))(input_img)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = ZeroPadding2D((1, 1))(model)
    model = Conv2D(64, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)

    model = ZeroPadding2D((1, 1))(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = ZeroPadding2D((1, 1))(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(model)

    model = ZeroPadding2D((1, 1))(model)
    model = Conv2D(256, (3, 3), activation='relu')(model)
    model = Reshape((512,5))(model)
    return model

# graph convolution #
def PI_convolution_layers(input_PI,A_in):
    model = GSConv(128,
                     activation='relu',
                           kernel_regularizer=l2(l2_reg),
                           use_bias=True)([input_PI, A_in])
    model = GSConv(5,
                     activation='relu',
                           kernel_regularizer=l2(l2_reg),
                           use_bias=True)([model, A_in])

    return model

# topological signature (PD) convolution #
def initial_rotation_PDs_null_layer(input_img):
    model = Exponentialtransformation(32,
                       activation='elu',
                       kernel_regularizer=l2(l2_reg),
                       deep_kernel = None,
                       initial_kernel=True)(input_img)
    model = Exponentialtransformation(5,
                                      activation='elu',
                                      kernel_regularizer=l2(l2_reg),
                                      deep_kernel=True,
                                      initial_kernel=None)(model)
    model = MaxPooling2D(pool_size=(1, 1), padding='same')(model)
    model = Reshape((5,5))(model)
    return model


def create_comb_convolution_layers(comb):
    model = Conv1D(filters=16,kernel_size=3,padding='same')(comb)
    model = LeakyReLU(alpha=0.01)(model)
    model = AveragePooling1D(pool_size=2, padding='same')(model)
    return model


# model #
sensor_input = Input(shape=(9, 20, 1))
initial_sensor_model = initial_sensor_convolution_layers(sensor_input)

PI_input = Input(shape=(400, 1))
initial_PI_model = PI_convolution_layers(PI_input, A_in)

conv = concatenate([initial_sensor_model, initial_PI_model], axis=1)
conv = create_comb_convolution_layers(conv)
conv = Flatten()(conv)

dense = Dense(512, activation='relu')(conv)
dense = BatchNormalization(axis=-1,center=True,scale=False)(dense)
dense = Dropout(0.5)(dense)
output = Dense(num_classes, activation='softmax')(dense)

model = Model(inputs=[sensor_input, PI_input, A_in], outputs=[output])

opt = optimizers.Adam(lr = learning_rate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()

model.fit([train_images, train_PI_null], train_labels,
              batch_size=32,
              epochs=epochs,
              verbose=1)

final_loss, final_acc = model.evaluate([test_images, test_PI_null], test_labels, verbose=1)
print("Final loss: {0:.6f}, final accuracy: {1:.6f}".format(final_loss, final_acc))
