# -*- coding: utf-8 -*-
"""
 @Author  : yangzh
 @Email   : 1725457378@qq.com
 @File    : ConvLSTM.py
"""
"""
Thank you for the excellent code at the bottom of keras. Here, the code at the bottom of keras is referenced and rewritten
"""

from keras.layers import ConvRNN2D, ConvLSTM2DCell
from keras.layers import regularizers, activations, initializers, constraints


class ConvLSTM2D(ConvRNN2D):

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
                 activity_regularizer = None,
                 kernel_constraint = None,
                 recurrent_constraint = None,
                 bias_constraint = None,
                 return_sequences = False,
                 go_backwards = False,
                 stateful = False,
                 dropout = 0.,
                 recurrent_dropout = 0.,
                 **kwargs):
        cell = ConvLSTM2DCell(filters = filters,
                              kernel_size = kernel_size,
                              strides = strides,
                              padding = padding,
                              data_format = data_format,
                              dilation_rate = dilation_rate,
                              activation = activation,
                              recurrent_activation = recurrent_activation,
                              use_bias = use_bias,
                              kernel_initializer = kernel_initializer,
                              recurrent_initializer = recurrent_initializer,
                              bias_initializer = bias_initializer,
                              unit_forget_bias = unit_forget_bias,
                              kernel_regularizer = kernel_regularizer,
                              recurrent_regularizer = recurrent_regularizer,
                              bias_regularizer = bias_regularizer,
                              kernel_constraint = kernel_constraint,
                              recurrent_constraint = recurrent_constraint,
                              bias_constraint = bias_constraint,
                              dropout = dropout,
                              recurrent_dropout = recurrent_dropout)
        super(ConvLSTM2D, self).__init__(cell,
                                         return_sequences = return_sequences,
                                         go_backwards = go_backwards,
                                         stateful = stateful,
                                         **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask = None, training = None, initial_state = None):
        return super(ConvLSTM2D, self).call(inputs,
                                            mask = mask,
                                            training = training,
                                            initial_state = initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

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
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint':
                      constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM2D, self).get_config( )
        del base_config['cell']
        return dict(list(base_config.items( )) + list(config.items( )))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
