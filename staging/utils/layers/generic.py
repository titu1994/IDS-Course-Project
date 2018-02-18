from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


class PriorScaling(Layer):

    def __init__(self, class_priors, **kwargs):
        super(PriorScaling, self).__init__(**kwargs)
        self.supports_masking = True

        self.prior_weights = np.asarray(class_priors)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

        self.prior_weights = 1. / self.prior_weights
        self.prior_weights = np.reshape(self.prior_weights, (1, -1))

        super(PriorScaling, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        y = x * self.prior_weights
        y = y / K.sum(y, axis=-1, keepdims=True)
        return y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.input_dim)
