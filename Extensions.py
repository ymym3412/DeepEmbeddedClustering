# coding:utf-8
from chainer.training import extension


class ChangeLearningRate(extension.Extension):
    """
    divide learning rate by 10 every call
    """
    def __init__(self, attr="lr", optimizer=None):
        self._attr = attr
        self.optimizer = optimizer

    def __call__(self, trainer):
        optimizer = self._get_optimizer(trainer)
        value = getattr(optimizer, self._attr) / 10
        self._update_optimizer(optimizer, value)


    def _get_optimizer(self, trainer):
        return self.optimizer or trainer.updater.get_optimizer("main")

    def _update_optimizer(self, optimizer, value):
        setattr(optimizer, self._attr, value)
