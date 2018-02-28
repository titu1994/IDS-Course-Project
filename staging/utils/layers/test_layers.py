import numpy as np

from keras.layers import Input, RNN
from keras.models import Model
from keras import backend as K

from staging.utils.layers import NestedLSTM
from staging.utils.layers import MinimalRNN
from staging.utils.layers import MultiplicativeLSTM
from staging.utils.layers import AttentionLSTM
from staging.utils.layers import NASCell
from staging.utils.layers import PriorScaling


def test_nested_lstm():
    K.clear_session()

    ip = Input(shape=(5, 10))
    x = NestedLSTM(8, depth=2)(ip)
    model = Model(ip, x)

    assert model.output_shape == (None, 8)


def test_minimal_rnn():
    K.clear_session()

    ip = Input(shape=(5, 10))
    x = MinimalRNN(8)(ip)
    model = Model(ip, x)

    assert model.output_shape == (None, 8)


def test_multiplicative_lstm():
    K.clear_session()

    ip = Input(shape=(5, 10))
    x = MultiplicativeLSTM(8)(ip)
    model = Model(ip, x)

    assert model.output_shape == (None, 8)


def test_attention_lstm():
    K.clear_session()

    ip = Input(shape=(5, 10))
    x = AttentionLSTM(8)(ip)
    model = Model(ip, x)

    assert model.output_shape == (None, 8)


def test_nas_cell():
    K.clear_session()

    ip = Input(shape=(5, 10))
    x = RNN(NASCell(8))(ip)
    model = Model(ip, x)

    assert model.output_shape == (None, 8)


def test_prior_scaling():
    K.clear_session()

    ip = Input(shape=(3,))
    x = PriorScaling(class_priors=[0.33, 0.33, 0.33])(ip)
    model = Model(ip, x)

    assert model.output_shape == (None, 3)
