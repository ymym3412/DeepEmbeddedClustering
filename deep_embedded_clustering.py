# coding: utf-8
import re

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


    def add_centroid(self, centroids):
        with self.init_scope():
            for i, centroid in enumerate(centroids):
                name = "u{}".format(i+1)
                initializer = LinearInitializer(centroid)
                self.add_param(name, centroid.shape, initializer=initializer)


class LinearInitializer(Initializer):
    def __init__(self, array, dtype=None):
        self.array = array.T
        super(LinearInitializer, self).__init__(dtype)

    def __call__(self, array):
        xp = cuda.get_array_module(array)
        k = xp.array([self.array], dtype=xp.float32)
        array[:] = k