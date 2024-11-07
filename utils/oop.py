import sys

# needed for optimizer and loss resolution
from torch import optim
from torch import nn


class Singleton(object):
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it
    def init(self, *args, **kwds):
        pass


def str_to_loss(classname: str):
    return getattr(sys.modules["torch.nn"], classname) 

def str_to_optimizer(classname: str):
    return getattr(sys.modules["torch.optim"], classname) 