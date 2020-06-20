from __future__ import absolute_import

import tensorflow as tf
import numpy as np
from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D
from keras import Input, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import RandomUniform
from keras.constraints import max_norm, non_neg, unit_norm, min_max_norm
import scipy.sparse as sp
from ops import sp_matrix_to_sp_tensor, transpose, mixed_mode_dot, filter_dot


_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Exponentialtransformation(Layer):

    def __init__(self,
                 channels,
                 activation=None,
                 initial_kernel=None,
                 deep_kernel=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Exponentialtransformation, self).__init__(**kwargs)
        self.initial_kernel = initial_kernel
        self.deep_kernel = deep_kernel
        self.channels = channels
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert input_shape[-1] >= 1
        self.kernel_mu0 = self.add_weight(shape=(1,),
                                      initializer=self.kernel_initializer,
                                      name='mu0_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.kernel_mu1 = self.add_weight(shape=(1,),
                                          initializer=self.kernel_initializer,
                                          name='mu1_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        self.kernel_sigma0 = self.add_weight(shape=(1,),
                                          initializer=self.kernel_initializer,
                                          name='sigma0_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        self.kernel_sigma1 = self.add_weight(shape=(1,),
                                          initializer=self.kernel_initializer,
                                          name='sigma1_kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        if self.initial_kernel:
            self.initial_kernel = self.add_weight(shape=(1, self.channels),
                                                  initializer=self.kernel_initializer,
                                                  name='initial_weight_kernel',
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        else:
            self.initial_kernel = None

        if self.deep_kernel:
            self.deep_kernel = self.add_weight(shape=(input_shape[-2], self.channels),
                                                  initializer=self.kernel_initializer,
                                                  name='deep_weight_kernel',
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint)
        else:
            self.deep_kernel = None

        self.nu = self.add_weight(shape=(1,),
                                             initializer=RandomUniform(minval=0., maxval=0.9),
                                             name='nu_kernel',
                                             regularizer=self.kernel_regularizer,
                                             constraint=min_max_norm(min_value=0.09, max_value=1))
        self.p = self.add_weight(shape=(1,),
                                             initializer=RandomUniform(minval=0., maxval=5.),
                                             name='p_kernel',
                                             regularizer=self.kernel_regularizer,
                                             constraint=min_max_norm(min_value=0., max_value=5.))

        self.built = True

    def call(self, inputs):

        if inputs.shape[-1] == 2:
            temp1 = (inputs[:, :, 0] - self.kernel_mu0)
            temp1 = tf.pow(tf.multiply(tf.pow(temp1,2), tf.pow(self.kernel_sigma0, 2)),self.p)

            nu = self.nu  # learnable nu parameter
            const = tf.constant(1., dtype=tf.float32)

            new_death_col = tf.where(tf.less(inputs[:, :, 1:], nu),
                                     tf.multiply(tf.log(tf.divide(inputs[:, :, 1:], nu)), nu) + nu,
                                     tf.multiply(inputs[:, :, 1:], const))

            temp2 = (new_death_col - self.kernel_mu1)
            temp2 = tf.pow(tf.multiply(tf.pow(temp2,2), tf.pow(self.kernel_sigma1, 2)),2)

            output = tf.exp(-temp1 - temp2)
            output = K.expand_dims(output, axis=-1)

            output = K.dot(output, self.initial_kernel)
            output = tf.nn.elu(output)
            output = K.expand_dims(output, axis=- 1)
        else:
            inputs = K.squeeze(inputs, axis=-1)
            output = K.dot(inputs, self.deep_kernel)
            output = tf.nn.elu(output)
            output = K.expand_dims(output, axis=- 1)

        return output

    def compute_output_shape(self, input_shape):
        if input_shape[-1] == 2:
            output_shape = input_shape[:-1] + (self.channels,) + (1,)
        else:
            output_shape = input_shape[:-2] + (self.channels,) + (1,)
        return output_shape

    def get_config(self):
        config = {
            'initial_kernel': self.initial_kernel,
            'deep_kernel': self.deep_kernel,
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(Exponentialtransformation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GraphConv(Layer):

    def __init__(self,
                 channels,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.channels = channels
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(input_dim, self.channels),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.channels,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = K.dot(features, self.kernel)
        output = filter_dot(fltr, output)

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))