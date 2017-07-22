# coding:utf-8
import chainer
import chainer.links as L
from chainer.datasets import mnist, tuple_dataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer import serializers
from chainer.training import extensions
from chainer.dataset import convert
from chainer.functions.loss.mean_squared_error import mean_squared_error
import cupy as cp
import numpy as np
import argparse

import Extensions
from StackedDenoisingAutoEncoder import StackedDenoisingAutoEncoder


def pretrain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batchsize', type=int, default=256)
    args = parser.parse_args()

    xp = np
    gpu_id = args.gpu
    seed = args.seed
    train, _ = mnist.get_mnist()
    train, _ = convert.concat_examples(train, device=gpu_id)
    batchsize = args.batchsize
    model = StackedDenoisingAutoEncoder(input_dim=train.shape[1])
    if chainer.cuda.available and args.gpu >= 0:
        xp = cp
        model.to_gpu(gpu_id)
    xp.random.seed(seed)

    # Layer-Wise Pretrain
    print("Layer-Wise Pretrain")
    for i, dae in enumerate(model.children()):
        print("Layer {}".format(i+1))
        train_tuple = tuple_dataset.TupleDataset(train, train)
        train_iter = iterators.SerialIterator(train_tuple, batchsize)
        clf = L.Classifier(dae, lossfun=mean_squared_error)
        clf.compute_accuracy = False
        if chainer.cuda.available and args.gpu >= 0:
            clf.to_gpu(gpu_id)
        optimizer = optimizers.MomentumSGD(lr=0.1)
        optimizer.setup(clf)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (50000, "iteration"), out="mnist_result")
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'elapsed_time']))
        trainer.extend(Extensions.ChangeLearningRate(), trigger=(20000, "iteration"))
        trainer.run()
        train = dae.encode(train).data

    # Finetuning
    print("fine tuning")
    with chainer.using_config("train", False):
        train, _ = mnist.get_mnist()
        train, _ = convert.concat_examples(train, device=gpu_id)
        train_tuple = tuple_dataset.TupleDataset(train, train)
        train_iter = iterators.SerialIterator(train_tuple, batchsize)
        model = L.Classifier(model, lossfun=mean_squared_error)
        model.compute_accuracy = False
        if chainer.cuda.available and args.gpu >= 0:
            model.to_gpu(gpu_id)
        optimizer = optimizers.MomentumSGD(lr=0.1)
        optimizer.setup(model)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (100000, "iteration"), out="mnist_result")
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['iteration', 'main/loss', 'elapsed_time']))
        trainer.extend(Extensions.ChangeLearningRate(), trigger=(20000, "iteration"))
        trainer.run()

    outfile = "StackedDenoisingAutoEncoder-seed{}.model".format(seed)
    serializers.save_npz(outfile, model.predictor)


if __name__ == '__main__':
    pretrain()