import keras
import numpy as np
from keras import backend as K

class ClusteringLayer(keras.layers.Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights

    def build(self, input_shape): 
        _shape = (self.n_clusters, ) + input_shape[1:]
        self.clusters = self.add_weight(shape=_shape,
                                        initializer='glorot_uniform', 
                                        name='clusters')
        self.sum_axis = tuple(range(2, len(input_shape) + 1))
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(ClusteringLayer, self).build(input_shape)


    def call(self, inputs, **kwargs): 
        def _pairwise_euclidean_distance(a, b):
            return K.sqrt(K.sum(K.pow(K.expand_dims(a, axis=1)-b, 2), axis=self.sum_axis))
            # return tf.norm(tf.expand_dims(a, axis=1) - b, 'euclidean', axis=(2,3))
        dist = _pairwise_euclidean_distance(inputs, self.clusters)
        q = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+1.0)/2.0)
        q = q/K.sum(q, axis=1, keepdims=True)
        return q

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def np_target_distribution(q):
    """If outside Keras scope"""
    weight = q**2 / q.sum(axis=0)
    return weight / weight.sum(axis=1, keepdims=True)


def target_distribution(q):
    """If inside keras scope"""
    weight = K.pow(q, 2) / K.sum(q, axis=0)
    return weight / K.sum(weight, axis=1, keepdims=True
