from keras.layers import Layer, InputSpec, Flatten
import tensorflow as tf
import numpy as np

class MILPooling(Layer):
    """
    MIL pooling layer that extracts the K-highest and K-lowest activations from a sequence (2nd dimension).
    """
    def __init__(self, kmax=1, kmin=0, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.kmax = kmax
        self.kmin = kmin

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.kmax + self.kmin, input_shape[2])

    def get_config(self):
        config = {'kmin': self.kmin, 'kmax': self.kmax}
        base_config = super(MILPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])

        # extract top_k
        # tf.nn.top_k returns two tensors [values, indices], keep values only
        top_k = tf.nn.top_k(shifted_input, k=self.kmax, sorted=False, name=None).values
        
        # extract bottom_k
        # using same tf.nn.top_k function on inverted values, then invert again to obtain original values
        if self.kmin > 0:
            bottom_k = tf.negative(tf.nn.top_k(tf.negative(shifted_input), k=self.kmin, sorted=False, name=None).values)
            
        # concatenate top_k and bottom_k in a single tensor
        out = tf.concat([top_k, bottom_k], -1) if self.kmin > 0 else top_k
        
        # swap last two dimensions again to obtain original shape
        out = tf.transpose(out, [0, 2, 1])

        return out
