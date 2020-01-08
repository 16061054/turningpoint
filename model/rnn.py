from typing import List, Tuple

import keras.backend as K
import keras.layers
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda
from keras.layers import CuDNNGRU, CuDNNLSTM, Dense, ReLU, Dropout, BatchNormalization, LSTM
from keras.layers import Bidirectional, Dense
from keras.models import Input, Model
from keras import initializers


class RNN(object):
    def __init__(self,
                 time_step=64,
                 fearure_dim=2,
                 hidden_size=256,
                 dropout_rate=0.0,
                 name='rnn'):
        self.name = name
        self.time_step = time_step
        self.fearure_dim = fearure_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def __call__(self, inputs):
        x = inputs
        x = CuDNNLSTM(self.hidden_size, return_sequences=True)(x)
        x = CuDNNLSTM(self.hidden_size, return_sequences=False)(x)
        return x

class RNN_SEPARATE_2(object):

    def __init__(self,
                 time_step=64,
                 fearure_dim=2,
                 hidden_size=256, # 128
                 dropout_rate=0.2,
                 name='rnn'):
        self.name = name
        self.time_step = time_step
        self.fearure_dim = fearure_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def __call__(self, inputs):
        x = inputs
        x_1 = Lambda(lambda x: x[:,:,0])(x)
        x_1 = Lambda(lambda x: K.expand_dims(x, axis=-1))(x_1)
        x_2 = Lambda(lambda x: x[:,:,1:])(x)

        h_2 = CuDNNLSTM(self.hidden_size, return_sequences=True)(x_2) # batch, seq2, dim
        h_2_firt = Lambda(lambda x: x[:,1:,])(h_2)
        h_2_last = Lambda(lambda x: x[:,-1,:])(h_2)

        h_1 = CuDNNLSTM(self.hidden_size, return_sequences=True)(x_1) # batch, seq1, dim
        h_1_first = Lambda(lambda x:x[:1:,:])(h_1)
        h_1_last = Lambda(lambda x: x[:,-1,:])(h_1) # batch, dim

        h = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([h_1, h_2]) # batch, seq, dim
        h = CuDNNLSTM(self.hidden_size, return_sequences=True)(h) # batch, seq, dim
        h = CuDNNLSTM(self.hidden_size, return_sequences=True)(h) # batch, seq, dim
        h_first = Lambda(lambda x: x[:,:-1,:])(h) # batch, seq-1, dim
        h_last = Lambda(lambda x: x[:,-1,:])(h) # batch, dim

        y = Lambda(lambda x:x[0]+x[1]+x[2])([h_1_last, h_2_last, h_last])
        return y



#------------------------------test model arch-----------------------------------------------------------
class RNN_SEPARATE_2_test(object):
    def __init__(self,
                 time_step=64,
                 fearure_dim=2,
                 hidden_size=256, # 128
                 dropout_rate=0.2,
                 name='rnn'):
        self.name = name
        self.time_step = time_step
        self.fearure_dim = fearure_dim
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def __call__(self, inputs):
        x = inputs
        x_1 = Lambda(lambda x: x[:,:,0])(x)
        x_1 = Lambda(lambda x: K.expand_dims(x, axis=-1))(x_1)
        x_2 = Lambda(lambda x: x[:,:,1:])(x)

        h_2 = CuDNNLSTM(self.hidden_size, return_sequences=True)(x_2) # batch, seq2, dim
        h_2_firt = Lambda(lambda x: x[:,1:,])(h_2)
        h_2_last = Lambda(lambda x: x[:,-1,:])(h_2)

        h_1 = CuDNNLSTM(self.hidden_size, return_sequences=True)(x_1) # batch, seq1, dim
        h_1_first = Lambda(lambda x:x[:1:,:])(h_1)
        h_1_last = Lambda(lambda x: x[:,-1,:])(h_1) # batch, dim

        h = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([h_1, h_2]) # batch, seq, dim
        h = CuDNNLSTM(self.hidden_size, return_sequences=True)(h) # batch, seq, dim
        h = CuDNNLSTM(self.hidden_size, return_sequences=True)(h) # batch, seq, dim
        h_first = Lambda(lambda x: x[:,:-1,:])(h) # batch, seq-1, dim
        h_last = Lambda(lambda x: x[:,-1,:])(h) # batch, dim

        # f0, att0 = ATT_SELF()(h_first) # batch, seq-1, dim
        # f1, att1 = ATT_ONE2MANY()([h_1_last, h_first]) # batch, seq-1, dim
        # f2, att2 = ATT_ONE2MANY()([h_2_last, h_first]) # batch, seq-1, dim

        # a1 = Lambda(lambda x:K.softmax(x[0] * x[1], axis=-1))([att0, att1]) # batch, seq-1
        # a2 = Lambda(lambda x:K.softmax(x[0] * x[1], axis=-1))([att0, att2]) # batch, seq-1

        # h_1_weight = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([h_first, a1]) # batch, dim
        # h_1_weight = Dense(self.hidden_size, use_bias=False)(h_1_weight) # # batch, dim
        # h_2_weight = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[1, 1]))([h_first, a2]) # batch, dim
        # h_2_weight = Dense(self.hidden_size, use_bias=False)(h_2_weight) # # batch, dim

        # h_weight = Lambda(lambda x: x[0]+x[1])([h_1_weight, h_2_weight])
        # h_weight = Dense(self.hidden_size, use_bias=True)(h_weight)
        # h_weight = ReLU()(h_weight)

        # y = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=-1))([h_weight, h_last])
        # y = Dense(self.hidden_size)(y)
        # y = ReLU()(y)

        y = Lambda(lambda x:x[0]+x[1]+x[2])([h_1_last, h_2_last, h_last])
        #y = Lambda(lambda x:K.stack([x[0], x[1], x[2]], axis=-1))([h_1_last, h_2_last, h_last]) # batch, dim, 3
        #y = Lambda(lambda x:K.permute_dimensions(x, [0, 2, 1]))(y) # batch, 3, dim
        #y, att_weight = ATT_SELF_2()(y)
        
        #y = h_last
        #y = ReLU()(y)

        return y


class ATT_SELF_1(Layer):
    """
    自注意力
    """

    def __init__(self, att_dim, **kwargs):
        self.init = initializers.get('uniform')
        self.att_dim = att_dim
        super(ATT_SELF_1, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], self.att_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        self.v = self.add_weight(shape=(self.att_dim, 1),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        self.trainable_weights = [self.W, self.v]
        super(ATT_SELF_1, self).build(input_shape)

    def call(self, x, mask=None):
        logits = K.dot(x, self.W)
        logits = K.tanh(logits)
        logits = K.dot(logits, self.v) # batch, seq, 1
        logits = K.squeeze(logits, axis=-1) # batch, seq
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        att_weights = ai / (K.sum(ai, axis=-1, keepdims=True) + K.epsilon()) # batch, seq
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1) # batch, dim
        return [result, att_weights]

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        step = input_shape[1]
        dim = input_shape[2]
        return [(batch, dim), (batch, step)]


class ATT_SELF_2(Layer):
    """
    点乘自注意力
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(ATT_SELF_2, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        super(ATT_SELF_2, self).build(input_shape)

    def call(self, x, mask=None):
        logits = K.dot(x, self.W) # batch, seq, 1
        logits = K.squeeze(logits, axis=-1) # batch, seq
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        att_weights = ai / (K.sum(ai, axis=-1, keepdims=True) + K.epsilon()) # batch, seq
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1) # batch, dim
        return [result, att_weights]

    def compute_output_shape(self, input_shape):
        batch = input_shape[0]
        step = input_shape[1]
        dim = input_shape[2]
        return [(batch, dim), (batch, step)]



class ATT_ONE2MANY_1(Layer):
    """
    加性注意力
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(ATT_ONE2MANY_1, self).__init__(**kwargs)

    def build(self, input_shape):
        one = input_shape[0]
        many = input_shape[1]
        assert len(one) == 2
        assert len(many) == 3

        self.W = self.add_weight(shape=(one[1], one[1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        super(ATT_ONE2MANY_1, self).build(input_shape)

    def call(self, x, mask=None):
        one = x[0] # batch, dim
        many = x[1] # bacth, seq, dim

        logits = K.dot(many, self.W)
        logits = K.tanh(logits)
        logits = K.batch_dot(logits, one, axes=[2, 1]) # batch, seq
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        att_weights = ai / (K.sum(ai, axis=-1, keepdims=True) + K.epsilon()) # batch, seq
        weighted_input = many * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1) # batch, dim
        return [result, att_weights]

    def compute_output_shape(self, input_shape):
        shape_one = input_shape[0]
        shape_many = input_shape[1]
        return [shape_one, shape_many[:2]]


#-------------------------------------------------------------------------------------------------

class ATT_ONE2MANY_2(Layer):
    """
    点乘注意力
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(ATT_ONE2MANY_2, self).__init__(**kwargs)

    def build(self, input_shape):
        one = input_shape[0]
        many = input_shape[1]
        assert len(one) == 2
        assert len(many) == 3

        super(ATT_ONE2MANY_2, self).build(input_shape)

    def call(self, x, mask=None):
        one = x[0] # batch, dim
        many = x[1] # bacth, seq, dim

        logits = K.batch_dot(many, one, axes=[2, 1]) # batch, seq
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        att_weights = ai / (K.sum(ai, axis=-1, keepdims=True) + K.epsilon()) # batch, seq
        weighted_input = many * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1) # batch, dim
        return [result, att_weights]

    def compute_output_shape(self, input_shape):
        shape_one = input_shape[0]
        shape_many = input_shape[1]
        return [shape_one, shape_many[:2]]


class ATT_ONE2MANY_3(Layer):
    """
    乘性注意力
    """

    def __init__(self, **kwargs):
        self.init = initializers.get('uniform')
        super(ATT_ONE2MANY_3, self).__init__(**kwargs)

    def build(self, input_shape):
        one = input_shape[0]
        many = input_shape[1]
        assert len(one) == 2
        assert len(many) == 3

        self.W = self.add_weight(shape=(one[1], one[1]),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        super(ATT_ONE2MANY_3, self).build(input_shape)

    def call(self, x, mask=None):
        one = x[0] # batch, dim
        many = x[1] # bacth, seq, dim

        logits = K.dot(many, self.W) # batch, seq, dim
        logits = K.batch_dot(logits, one, axes=[2, 1]) # batch, seq
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        att_weights = ai / (K.sum(ai, axis=-1, keepdims=True) + K.epsilon()) # batch, seq
        weighted_input = many * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1) # batch, dim
        return [result, att_weights]

    def compute_output_shape(self, input_shape):
        shape_one = input_shape[0]
        shape_many = input_shape[1]
        return [shape_one, shape_many[:2]]