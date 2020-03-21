# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : ConvLSTM2DCell.py
"""
"""
Thank you for the excellent code at the bottom of keras. Here, the code at the bottom of keras is referenced and rewritten
"""
import warnings

from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Subtract, Permute, multiply, ConvLSTM2DCell, GlobalAveragePooling2D, Dense, Reshape
from keras.layers import regularizers, activations, initializers, constraints
from keras.layers.recurrent import _generate_dropout_mask
from keras.utils import conv_utils


class ConvLSTM2DCell(Layer):

    def __init__(self, filters,
                 kernel_size,
                 strides = (1, 1),
                 padding = 'valid',
                 data_format = None,
                 dilation_rate = (1, 1),
                 activation = 'tanh',
                 recurrent_activation = 'hard_sigmoid',
                 use_bias = True,
                 kernel_initializer = 'glorot_uniform',
                 recurrent_initializer = 'orthogonal',
                 bias_initializer = 'zeros',
                 unit_forget_bias = True,
                 kernel_regularizer = None,
                 recurrent_regularizer = None,
                 bias_regularizer = None,
                 kernel_constraint = None,
                 recurrent_constraint = None,
                 bias_constraint = None,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 **kwargs):
        super(ConvLSTM2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = K.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        if K.backend( ) == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters)
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)

        self.kernel = self.add_weight(shape = kernel_shape,
                                      initializer = self.kernel_initializer,
                                      name = 'kernel',
                                      regularizer = self.kernel_regularizer,
                                      constraint = self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape = recurrent_kernel_shape,
            initializer = self.recurrent_initializer,
            name = 'recurrent_kernel',
            regularizer = self.recurrent_regularizer,
            constraint = self.recurrent_constraint)
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        initializers.Ones( )((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape = (self.filters * 4,),
                                        name = 'bias',
                                        initializer = bias_initializer,
                                        regularizer = self.bias_regularizer,
                                        constraint = self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :, :, :self.filters]
        self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.filters]
        self.kernel_f = self.kernel[:, :, :, self.filters: self.filters * 2]
        self.recurrent_kernel_f = (
            self.recurrent_kernel[:, :, :, self.filters: self.filters * 2])
        self.kernel_c = self.kernel[:, :, :, self.filters * 2: self.filters * 3]
        self.recurrent_kernel_c = (
            self.recurrent_kernel[:, :, :, self.filters * 2: self.filters * 3])
        self.kernel_o = self.kernel[:, :, :, self.filters * 3:]
        self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.filters * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.filters]
            self.bias_f = self.bias[self.filters: self.filters * 2]
            self.bias_c = self.bias[self.filters * 2: self.filters * 3]
            self.bias_o = self.bias[self.filters * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, training = None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training = training,
                count = 4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[1]),
                self.recurrent_dropout,
                training = training,
                count = 4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        x_i = self.input_conv(inputs_i, self.kernel_i, self.bias_i,
                              padding = self.padding)
        x_f = self.input_conv(inputs_f, self.kernel_f, self.bias_f,
                              padding = self.padding)
        x_c = self.input_conv(inputs_c, self.kernel_c, self.bias_c,
                              padding = self.padding)
        x_o = self.input_conv(inputs_o, self.kernel_o, self.bias_o,
                              padding = self.padding)
        h_i = self.recurrent_conv(h_tm1_i,
                                  self.recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f,
                                  self.recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c,
                                  self.recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o,
                                  self.recurrent_kernel_o)

        i = self.recurrent_activation(self.se_block((x_i + h_i), ratio = 8, action = "+"))
        f = self.recurrent_activation(self.se_block((x_f + h_f), ratio = 8, action = "-"))
        c = f * c_tm1 + i * self.activation(x_c + h_c)

        r = activations.exponential( )(i * self.activation(x_c + h_c))
        r_o = self.recurrent_conv(r, self.recurrent_kernel_o)

        o = self.recurrent_activation(x_o + h_o + r_o)
        h = o * self.activation(c)

        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True

        return h, [h, c]

    def input_conv(self, x, w, b = None, padding = 'valid'):
        conv_out = K.conv2d(x, w, strides = self.strides,
                            padding = padding,
                            data_format = self.data_format,
                            dilation_rate = self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format = self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides = (1, 1),
                            padding = 'same',
                            data_format = self.data_format)
        return conv_out

    def se_block(self, input_feature, ratio = 8, action = "+"):
        """Contains the implementation of Squeeze-and-Excitation(SE) block.
        	As described in https://arxiv.org/abs/1709.01507.
        	"""
        channel_axis = 1 if K.image_data_format( ) == "channels_first" else -1
        channel = input_feature._keras_shape[channel_axis]

        se_feature = GlobalAveragePooling2D( )(input_feature)
        se_feature = Reshape((1, 1, channel))(se_feature)
        assert se_feature._keras_shape[1:] == (1, 1, channel)
        se_feature = Dense(channel // ratio,
                           activation = 'relu',
                           kernel_initializer = 'he_normal',
                           use_bias = True,
                           bias_initializer = 'zeros')(se_feature)
        assert se_feature._keras_shape[1:] == (1, 1, channel // ratio)
        se_feature = Dense(channel,
                           activation = 'sigmoid',
                           kernel_initializer = 'he_normal',
                           use_bias = True,
                           bias_initializer = 'zeros')(se_feature)
        assert se_feature._keras_shape[1:] == (1, 1, channel)
        if K.image_data_format( ) == 'channels_first':
            se_feature = Permute((3, 1, 2))(se_feature)
        if action == "+":
            se_feature = multiply([input_feature, se_feature])
        elif action == "-":
            se_feature = Subtract([input_feature, se_feature])
        return se_feature

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM2DCell, self).get_config( )
        return dict(list(base_config.items( )) + list(config.items( )))
