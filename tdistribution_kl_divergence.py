# coding:utf-8
from itertools import chain, repeat

import chainer
from chainer import cuda


class TdistributionKLDivergence(chainer.Function):

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        z, us = inputs[0], xp.array(inputs[1:], dtype=xp.float32)

        dist_matrix = xp.linalg.norm(xp.vstack(chain.from_iterable(map(lambda v: repeat(v, us.shape[0]), z))) - xp.vstack(repeat(us, z.shape[0])), axis= 1).reshape(z.shape[0], us.shape[0])
        q_matrix = (self.tdistribution_kernel(dist_matrix).T / self.tdistribution_kernel(dist_matrix).sum(axis=1)).T
        p_matrix = self.compute_pmatrix(q_matrix)
        kl_divergence = (p_matrix * (xp.log(p_matrix) - xp.log(q_matrix))).sum()
        return xp.array(kl_divergence),


    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        z, us = inputs[0], xp.array(inputs[1:], dtype=xp.float32)
        gloss, = grad_outputs
        gloss = gloss / z.shape[0]

        # z
        norms = xp.vstack(chain.from_iterable(map(lambda v: repeat(v, us.shape[0]), z))) - xp.vstack(repeat(us, z.shape[0]))
        z_norm_matrix = norms.reshape(z.shape[0], us.shape[0], z.shape[1])

        dist_matrix = xp.linalg.norm(norms, axis= 1).reshape(z.shape[0], us.shape[0])
        q_matrix = (self.tdistribution_kernel(dist_matrix).T / self.tdistribution_kernel(dist_matrix).sum(axis=1)).T
        p_matrix = self.compute_pmatrix(q_matrix)

        gz = 2 * ((((p_matrix - q_matrix) * self.tdistribution_kernel(dist_matrix)) * z_norm_matrix.transpose(2,0,1)).transpose(1,2,0)).sum(axis=1) * gloss
        gus = -2 * ((((p_matrix - q_matrix) * self.tdistribution_kernel(dist_matrix)) * z_norm_matrix.transpose(2,0,1)).transpose(1,2,0)).sum(axis=0) * gloss

        g = [gz]
        g.extend(gus)
        grads = tuple(g)
        return grads


    def tdistribution_kernel(self, norm):
        xp = cuda.get_array_module(norm)
        return xp.power((1 + norm), -1)


    def compute_pmatrix(self, q_matrix):
        xp = cuda.get_array_module(q_matrix)
        fj = q_matrix.sum(axis=0)
        matrix = xp.power(q_matrix, 2) / fj
        p_matrix = (matrix.T / matrix.sum(axis=1)).T
        return p_matrix


def tdistribution_kl_divergence(z, us):

    return TdistributionKLDivergence()(z, *us)