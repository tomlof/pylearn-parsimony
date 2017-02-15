# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.neural` module contains classes for neural
networks.

Loss functions should be stateless. Loss functions may be shared and copied
and should therefore not hold anything that cannot be recomputed the next time
it is called.

Created on Tue Feb 14 22:49:28 2017

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc
import copy
import types
from six import with_metaclass

import numpy as np

try:
    from . import properties  # Only works when imported as a package.
except ValueError:
    import parsimony.functions.properties as properties  # Run as a script.
#try:
#    from . import combinedfunctions  # Only works when imported as a package.
#except ValueError:
#    import parsimony.functions.combinedfunctions as combinedfunctions  # Run as a script.
#try:
#    from .multiblock import losses as multiblocklosses  # Only works when imported as a package.
#except ValueError:
#    import parsimony.functions.multiblock.losses as multiblocklosses  # Run as a script.
#try:
#    from .multiblock import properties as multiblockprops  # Only works when imported as a package.
#except ValueError:
#    import parsimony.functions.multiblock.properties as multiblockprops  # Run as a script.
#import parsimony.utils as utils
#import parsimony.utils.consts as consts


__all__ = ["BaseNode"]


class BaseNode(with_metaclass(abc.ABCMeta,
                              properties.Function,
                              properties.Derivative)):
    """This is the base class for all nodes in the network.
    """
    def __init__(self):

        pass

    def reset(self):
        """Free any cached computations from previous use of this Function.

        From the interface "Function".
        """
        pass


class InputNode(BaseNode):
    """This is the base class for all nodes in the network that are input
    nodes.
    """
    def __init__(self, x):
        self.x = x

    def f(self, x):
        return self.x

    def derivative(self, x):
        return 0.0  # These do not depend on the weights.


class ActivationNode(BaseNode):
    """This is the base class for all nodes in the network that have activation
    functions.
    """
    pass


class OutputNode(BaseNode):
    """This is the base class for all nodes in the network that are output
    nodes.
    """
    pass


class IdentityNode(ActivationNode):
    """A node where the activation function is the identity:

        f(x) = x.
    """
    def f(self, x):
        return x

    def derivative(self, x):
        return 1.0


class LogisticNode(ActivationNode):
    """A node where the activation function is the logistic function (soft
    step):

        f(x) = 1 / (1 + exp(-x)).
    """
    def f(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        f = self.f(x)
        return f * (1.0 - f)


class TanhNode(ActivationNode):
    """A node where the activation function is the hyperbolic tangent:

        f(x) = tanh(x) = (2 / (1 + exp(-2x))) - 1.
    """
    def f(self, x):
        return (2.0 / (1.0 + np.exp(-2.0 * x))) - 1.0

    def derivative(self, x):
        f = self.f(x)
        return 1.0 - (f ** 2.0)


class ReluNode(ActivationNode):
    """A node where the activation function is a rectified linear unit:

        f(x) = max(0, x).
    """
    def f(self, x):
        if x < 0.0:
            return 0.0
        else:
            return x

    def derivative(self, x):
        if x < 0.0:
            return 0.0
        else:
            return 1.0


class SquaredSumOutput(OutputNode):
    """An output node with the loss

        f(x) = (1 / 2) * (y - x)².
    """
    def __init__(self, y):
        self.y = y

    def f(self, x):
        d = self.y - x
        return 0.5 * d * d

    def derivative(self, x):
        return -(self.y - x)


class LogisticOutput(OutputNode):
    """An output node with the loss

        f(x) = 1 / (1 + exp(-x)).
    """
    def __init__(self, y):
        self.y = y

    def f(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        f = self.f(x)
        return f * (1.0 - f)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
