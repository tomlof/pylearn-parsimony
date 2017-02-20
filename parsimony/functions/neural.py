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
           "BaseLayer", "InputLayer", "HiddenLayer", "OutputLayer",
           "BaseNode", "InputNode", "ActivationNode", "OutputNode",
           "IdentityNode", "LogisticNode", "TanhNode", "ReluNode",
           "BaseLoss", "SquaredSumLoss", "BinaryCrossEntropyLoss",
           "CategoricalCrossEntropyLoss"]


def _init_weights(num_output, num_input):
    return (0.01 / np.sqrt(num_output + num_input)) * \
        np.random.randn(num_output, num_input)


class BaseNetwork(with_metaclass(abc.ABCMeta,
                                 properties.Function,
                                 properties.Gradient)):
    """This is the base class for all neural networks.
    """
    def __init__(self, loss, input_size=None, cache_activations=True):  # , output_size=None):

        self._input = InputLayer(num_nodes=input_size)
        self._layers = []
#        self._output = OutputLayer(num_nodes=output_size)
        self._loss = loss

        self.cache_activations = bool(cache_activations)

        self._input.connect_next(None)
#        self._output.connect_prev(self._input)

    def reset(self):
        # self._input = InputLayer(num_nodes=self._input.num_nodes())
        self._input.connect_next(None)
        self._layers = []
#        self._output = OutputLayer(num_nodes=self._output.num_nodes())

    def add_layer(self, layer):

#        self._output.connect_prev(None)

        if len(self._layers) == 0:

            self._input.connect_next(layer)  # Connect input layer to this
            layer.connect_prev(self._input)  # Connect this layer to input
        else:
            self._layers[-1].connect_next(layer)  # Connect last layer to this
            layer.connect_prev(self._layers[-1])  # Connect this layer to last

        layer.connect_next(None)  # Make last layer

        self._layers.append(layer)
#        self._output.connect_prev(self._layers[-1])

#        return len(self._layers) - 1  # Return index of the layer

    def set_target(self, target):

        self._loss.set_target(target)

#    def set_output(self, output_size):
#        self.output = OutputLayer(num_input_nodes=-1,
#                                  num_output_nodes=output_size)

    def forward(self):
        pass

    def backward(self):
        pass


class FeedForwardNetwork(BaseNetwork):

    def f(self, x):
        x = x.T  # The network needs the samples in the rows.

        y = self.forward(x)

        E = self._loss.f(y)

        return E

    def grad(self, x):

#        if not self.cache_activations:
#            self.f(x)

        pass  # Implement!

    def forward(self, x):

        x = self._input.forward(x)  # Should just be the identity.

        num_layers = len(self._layers)
        for i in range(num_layers):
            layer = self._layers[i]
            x = layer.forward(x)

    def backward(self, x):

        target = self._loss.get_target()

        if len(self._layers) == 0:
            delta_output = self._loss.derivative(x)
        else:
            delta_output = self._loss.derivative(self._layers[-1].get_activation())


class BaseLayer(with_metaclass(abc.ABCMeta)):
    """This is the base class for all layers.
    """
    def __init__(self, num_nodes=None, nodes=None, weights=None, biases=None):

        self._num_nodes = num_nodes
        self._nodes = nodes
        if weights is not None:
            self._weights = np.asarray(weights)
        else:
            self._weights = weights
        if biases is not None:
            self._biases = np.asarray(biases)
        else:
            self._biases = biases

        self._all_same = True
        if isinstance(nodes, list):
            self._all_same = False

        self._prev_layer = None
        self._next_layer = None

    def reset(self):
        self._cached_Wx_b = None

    def num_nodes(self):

        return self._num_nodes

    def connect_next(self, layer):

        self._next_layer = layer

    def connect_prev(self, layer):

        self._prev_layer = layer

        if layer is None:
            self._weights = None
        elif self._weights is None:
            self._weights = _init_weights(self.num_nodes(), layer.num_nodes())

    def forward(self, inputs):

        Wx = np.dot(self._weights, inputs)
        if self._biases is not None:
            self._cached_Wx_b = Wx + self._biases
        else:
            self._cached_Wx_b = Wx

        if self._all_same:
            outputs = self._nodes.f(self._cached_Wx_b)
        else:
            outputs = np.zeros((self._num_output_nodes, 1))
            for i in range(self._num_output_nodes):
                outputs[i] = self._nodes[i].f(self._cached_Wx_b[i])

        return outputs

    def backward(self, grads):

        pass  # TODO: implement!!


class InputLayer(BaseLayer):
    """Represents an input layer.
    """
    def __init__(self, num_nodes=None):

        super(InputLayer, self).__init__(num_nodes=num_nodes,
                                         nodes=IdentityNode(),
                                         weights=1)

    def connect_prev(self, layer):

        raise ValueError("Cannot add a previous layer to the input layer!")

    def forward(self, inputs):

        return inputs


class HiddenLayer(BaseLayer):
    """Represents a hidden layer.
    """
    pass


#class OutputLayer(BaseLayer):
#    """Represents an output layer.
#    """
#    def __init__(self, num_nodes=None, nodes=None, loss=None, weights=None):
#
#        super(OutputLayer, self).__init__(num_nodes=num_nodes,
#                                          nodes=OutputNode(),
#                                          weights=weights)
#
#        self.set_loss(loss)
#
#    def backward(self, grads):
#
#        return grads  # Delta for the output layer (identity function)


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


class ActivationNode(with_metaclass(abc.ABCMeta, BaseNode)):
    """This is the base class for all nodes in the network that have activation
    functions.
    """
    pass


#class OutputNode(ActivationNode):
#    """This is the base class for all nodes in the network that are output
#    nodes.
#    """
#    def f(self, x):
#        return x
#
#    def derivative(self, x):
#        if isinstance(np.ndarray):
#            return np.ones(x.shape)
#        else:
#            return 1.0


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

        self.set_target(target)

    def set_target(self, target):

        if target is not None:
            if len(target.shape) != 2:
                raise ValueError("The target must be of shape 1-by-k.")
            if target.shape[0] != 1:
                target = target.reshape((1, np.prod(target.shape)))

        self.target = target

    def get_target(self):

        return self.target


class SquaredSumLoss(BaseLoss):
    """A squared error loss function

        f(x) = (1 / 2) * \sum_{i=1}^n (x_i - t_i)².
    """
    def f(self, x):

        x = x.T  # The network assumes the samples are in the rows.

        n = float(x.size)

        return (0.5 * n) * np.sum((x - self.target) ** 2.0)

    def derivative(self, x):

        x = x.T  # The network assumes the samples are in the rows.

        n = float(x.size)

        return (x - self.target) / n


class BinaryCrossEntropyLoss(BaseLoss):
    """A set of independent cross-entropy losses, with function

        f(x) = \sum_{i=1}^n -t_i * log(x_i) - (1 - t_i) * log(1 - x_i).
    """
    def f(self, x):
        x = x.T  # The network assumes the samples are in the rows.

        return -np.sum(np.multiply(self.target, np.log(x)) +
                       np.multiply(1.0 - self.target, np.log(1.0 - x)))

    def derivative(self, x):
        x = x.T  # The network assumes the samples are in the rows.

        return np.divide(x - self.target, np.multiply(x, 1.0 - x))


class CategoricalCrossEntropyLoss(BaseLoss):
    """A set of dependent outputs in a single cross-entropy loss, with function

        f(x) = -\sum_{i=1}^n t_i * log(x_i).
    """
    def f(self, x):
        x = x.T  # The network assumes the samples are in the rows.

        return -np.sum(np.multiply(self.target, np.log(x)))

    def derivative(self, x):
        x = x.T  # The network assumes the samples are in the rows.

        return -np.divide(self.target, x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
