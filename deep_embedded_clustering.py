# coding: utf-8
import re
from itertools import chain, repeat

import chainer
import chainer.functions as F
from chainer import Initializer
from chainer import cuda


class DeepEmbeddedClustering(chainer.ChainList):
    def __init__(self, chains):
        l1, l2, l3, l4 = chains
        super(DeepEmbeddedClustering, self).__init__(
            l1,
            l2,
            l3,
            l4
        )

    def __call__(self, x):
        # encode
        for dae in self.children():
            x = dae.encode(x)
        return x


    def get_centroids(self):
        p = r"u[0-9]{0,}"
        param_tuple = tuple((param for param in self.params() if re.match(p, param.name)))
        concat_param = F.vstack(param_tuple)
        return concat_param


    def add_centroids(self, centroids):
        for i, centroid in enumerate(centroids):
            name = "u{}".format(i+1)
            initializer = LinearInitializer(centroid)
            self.add_param(name, centroid.shape, initializer=initializer)


    def predict_label(self, x):
        z = self(x).data
        centroids = self.get_centroids().data
        xp = cuda.get_array_module(z)
        dist_matrix = xp.linalg.norm(xp.vstack(chain.from_iterable(map(lambda v: repeat(v, centroids.shape[0]), z)))\
                                     - xp.vstack(repeat(centroids, z.shape[0])), axis=1).reshape(z.shape[0], centroids.shape[0])
        q_matrix = (xp.power((1 + dist_matrix), -1).T / xp.power((1 + dist_matrix), -1).sum(axis=1)).T
        return xp.argmax(q_matrix, axis=1)



class LinearInitializer(Initializer):
    def __init__(self, array, dtype=None):
        self.array = array.T
        super(LinearInitializer, self).__init__(dtype)

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        array[:] = xp.array([self.array], dtype=xp.float32)