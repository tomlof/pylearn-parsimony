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


__all__ = ["BaseNetwork", "FeedForwardNetwork",

           "BaseNode"]


def init_weights(num_output, num_input):
    return (0.01 / np.abs(num_output - num_input)) * \
        np.random.randn(num_output, num_input)


class BaseNetwork(with_metaclass(abc.ABCMeta,
                                 properties.Function,
                                 properties.Gradient)):
    """This is the base class for all neural networks.
    """
    def __init__(self, input_size, loss):

        self._layers = [InputLayer(input_size)]

        self.loss = loss

    def add_layer(self, layer):

        self._layers[-1].connect_next(layer)  # Connect last layer to this
        layer.connect_prev(self._layers[-1])  # Connect this layer to last

        self._layers.append(layer)

#        return len(self._layers) - 1  # Return index of the layer


class FeedForwardNetwork(BaseNetwork):

    def __init__(self, input_size, loss):

        self._layers = [InputLayer(input_size)]

        self.loss = loss

    def f(self, x):
        num_layers = len(self._layers)
        for i in range(num_layers):
            layer = self._layers[i]
            x = layer.forward(x)

        loss = self.loss.f(x)

        return loss

    def grad(self, x):
        pass  # Implement!


class BaseLayer(with_metaclass(abc.ABCMeta), object):
    """This is the base class for all layers.
    """
    def __init__(self, num_input_nodes=1, num_output_nodes=1, nodes=None,
                 weights=None):

        self._nodes = nodes
        self._num_input_nodes = num_input_nodes
        self._num_output_nodes = num_output_nodes

        if weights is None:
            self._weights = init_weights(num_output_nodes, num_input_nodes)
        else:
            self._weights = np.asarray(weights)

        self._all_same = True
        if isinstance(nodes, list):
            self._all_same = False

        self._prev_layer = None
        self._next_layer = None

    def connect_next(self, layer):

        if self._num_output_nodes == layer._num_input_nodes:
            self._next_layer = layer
        else:
            raise ValueError("Node mismatch! Number of output nodes %d, "
                             "number of input nodes %d!"
                             % (self._num_output_nodes,
                                layer._num_input_nodes))

    def connect_prev(self, layer):

        if self._num_input_nodes == layer._num_output_nodes:
            self._prev_layer = layer
        else:
            raise ValueError("Node mismatch! Number of input nodes %d, "
                             "number of output nodes %d!"
                             % (self._num_input_nodes,
                                layer._num_output_nodes))

    def forward(self, inputs):

        Wx = np.dot(self._weights, inputs)
        if self._all_same:
            outputs = self._nodes.f(Wx)
        else:
            outputs = np.zeros((self._num_output_nodes, 1))
            for i in range(self._num_output_nodes):
                outputs[i] = self._nodes[i].f(Wx[i])

        return outputs

    def backward(self, grads):

        pass  # TODO: implement!!


class InputLayer(BaseLayer):
    """Represents an input layer.
    """
    def __init__(self, num_output_nodes=1):

        super(InputLayer, self).__init__(num_input_nodes=0,
                                         num_output_nodes=num_output_nodes,
                                         nodes=IdentityNode(),
                                         weights=1)

    def connect_prev(self, layer):

        raise ValueError("Cannot add a previous layer to the input layer!")

    def forward(self, inputs):

        return inputs


class HiddenLayer(BaseLayer):
    """Represents a hidden layer.
    """
    def __init__(self, num_input_nodes=1, num_output_nodes=1, nodes=None):

        super(BaseLayer, self).__init__(num_input_nodes=num_input_nodes,
                                        num_output_nodes=num_output_nodes,
                                        nodes=nodes)


class OutputLayer(HiddenLayer):
    """Represents an output layer.
    """
    pass


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


class OutputNode(ActivationNode):
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
        if isinstance(np.ndarray):
            return np.ones(x.shape)
        else:
            return 1.0


class LogisticNode(ActivationNode):
    """A node where the activation function is the logistic function (soft
    step):

        f(x) = 1 / (1 + exp(-x)).
    """
    def f(self, x):
        if isinstance(np.ndarray):
            return np.reciprocal(1.0 + np.exp(-x))
        else:
            return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        f = self.f(x)
        if isinstance(np.ndarray):
            return np.multiply(f, 1.0 - f)
        else:
            return f * (1.0 - f)


class TanhNode(ActivationNode):
    """A node where the activation function is the hyperbolic tangent function:

        f(x) = tanh(x) = (2 / (1 + exp(-2x))) - 1.
    """
    def f(self, x):
        if isinstance(np.ndarray):
            return 2.0 * np.reciprocal(1.0 + np.exp(-2.0 * x)) - 1.0
        else:
            return (2.0 / (1.0 + np.exp(-2.0 * x))) - 1.0

    def derivative(self, x):
        f = self.f(x)
        return 1.0 - (f ** 2.0)


class ReluNode(ActivationNode):
    """A node where the activation function is a rectified linear unit:

        f(x) = max(0, x).
    """
    def f(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.sign(np.maximum(0, x))


class BaseLoss(with_metaclass(abc.ABCMeta,
                              properties.Function,
                              properties.Derivative)):
    """This is the base class for all losses in the network.
    """
    def __init__(self, target=None):
        if target is not None:
            self.set_target(target)

    def set_target(self, target):
        self.target = target


class SquaredSumLoss(BaseLoss):
    """A squared error loss function

        f(x) = (1 / 2) * \sum_{i=1}^n (x_i - t_i)².
    """
    def f(self, x):
        return (1.0 / 2.0) * np.sum((x - self.target) ** 2.0)

    def derivative(self, x):
        return (x - self.target)


class BinaryCrossEntropyLoss(BaseLoss):
    """A set of independent cross-entropy losses, with function

        f(x) = \sum_{i=1}^n -t_i * log(x_i) - (1 - t_i) * log(1 - x_i).
    """
    def f(self, x):
        return -np.sum(np.multiply(self.target, np.log(x)) +
                       np.multiply(1.0 - self.target, np.log(1.0 - x)))

    def derivative(self, x):
        return np.divide(x - self.target, np.multiply(x, 1.0 - x))


class CategoricalCrossEntropyLoss(BaseLoss):
    """A set of dependent outputs in a single cross-entropy loss, with function

        f(x) = -\sum_{i=1}^n t_i * log(x_i).
    """
    def f(self, x):
        return -np.sum(np.multiply(self.target, np.log(x)))

    def derivative(self, x):
        return -np.divide(self.target, x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
