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
from parsimony.utils import check_arrays
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
    def __init__(self, X, y, output, loss):
        """The base class for neural networks.

        Parameters
        ----------
        X : numpy.ndarray, shape (n-by-p)
            The training data.

        y : numpy.ndarray, shape (n-by-q)
            The example outputs.

        output : parsimony.neural.BaseNode or list of parsimony.neural.BaseNode
            The nodes of the ourput layer. If passed a BaseNode, then all nodes
            of the output layer are of this type; if passed a list of
            BaseNodes, then each node's type is given by the corresponding
            element in the list.

        loss : parsimony.neural.BaseLoss
            The loss function of the network.
        """
        X, y = check_arrays(X, y)

        self.X = X
        self.y = y
        self._input = InputLayer(num_nodes=X.shape[1])
        self._layers = []
        self._output = OutputLayer(loss, num_nodes=y.shape[1], nodes=output)

        loss.set_target(y.T)  # Note: The network outputs column vectors

        self.reset()

    def reset(self):
        self._input.connect_next(self._output)
        self._output.connect_prev(self._input)
        self._layers = []

    def add_layer(self, layer):

        if len(self._layers) == 0:
            self._input.connect_next(layer)  # Connect input to this layer
            layer.connect_prev(self._input)  # Connect this layer to input
        else:
            self._layers[-1].connect_next(layer)  # Connect last layer to this
            layer.connect_prev(self._layers[-1])  # Connect this layer to last

        layer.connect_next(self._output)  # Connect this layer to output
        self._output.connect_prev(layer)  # Connect output to this layer

        self._layers.append(layer)

    def get_weights(self):

        weights = [0] * (len(self._layers) + 1)  # Num layers + output layer
        for i in range(len(self._layers)):
            weights[i] = self._layers[i].get_weights()

        weights[-1] = self._output.get_weights()

        return weights

    def set_weights(self, weights):

        for i in range(len(self._layers)):
            self._layers[i].set_weights(weights[i])

        self._output.set_weights(weights[-1])

    @abc.abstractmethod
    def _forward(self, X):
        raise NotImplementedError('Abstract method "_forward" must be '
                                  'specialised!')

    @abc.abstractmethod
    def _backward(self, y):
        raise NotImplementedError('Abstract method "_backward" must be '
                                  'specialised!')


class FeedForwardNetwork(BaseNetwork):

    def f(self, W):

        self.set_weights(W)

        output = self._output._forward(self.X.T)
        E = self._output.get_loss().f(output)

        return E

    def grad(self, W):

        self.set_weights(W)

        output = self._forward(self.X.T)  # Compute network output
        self._backward(output)  # Compute deltas (last delta is from loss)

        n_layers = len(self._layers)
        grad = [0.0] * (n_layers + 1)
        for i in range(n_layers):
#            ai = self._layers[i].get_activation()  # ai = gi(Wi * aj)
#            delta = self._layers[i].get_delta()  # Delta from layer above
#            grad[i] = np.dot(delta, ai.T)
            grad[i] = self._layers[i].get_grad()

        grad[-1] = self._output.get_grad()

        return grad

    def approx_grad(self, W, eps=1e-4):
        """Numerical approximation of the gradient.

        Parameters
        ----------
        W : list of numpy.ndarray, list of shape (num_samples, num_output)
            The point at which to evaluate the gradient.

        eps : float
            Positive float. The precision of the numerical solution. Smaller
            is better, but too small may result in floating point precision
            errors.
        """
        grad = [0] * len(W)
        num_layers = len(self._layers)
        for l in range(num_layers + 1):
            Wi = W[l].ravel()
            gradl = np.zeros(W[l].shape).ravel()

            p = Wi.size
#            if isinstance(self, (Penalty, Constraint)):
#                start = self.penalty_start
#            else:
            start = 0
            for i in range(start, p):
                Wi[i] -= eps
                loss1 = self.f(W)
                Wi[i] += 2.0 * eps
                loss2 = self.f(W)
                Wi[i] -= eps

                gradl[i] = (loss2 - loss1) / (2.0 * eps)

            grad[l] = gradl.reshape(W[l].shape)

        return grad

    def _forward(self, X):
        """Performs a forward pass recursively from the output layer.

        Parameters
        ----------
        y : numpy.ndarray, shape (num_output_nodes, num_samples)
            The output examples.
        """
#        num_layers = len(self._layers)
#        if num_layers == 0:
        y = self._output._forward(X)  # Last layer's output
#        else:
#            last_layer = self._layers[-1]
#            y = last_layer._forward(X)  # Last layer's output

        return y

    def _backward(self, y):
        """Performs a backwards pass recursively from the first layer.

        Parameters
        ----------
        y : numpy.ndarray, shape (num_output_nodes, num_samples)
            The output examples.
        """
#        n_layers = len(self._layers)
#        delta = [0.0] * (n_layers + 1)
#
#        delta[n_layers] = self._loss.derivative(y)
#
#        if n_layers == 0:
#            delta[n_layers - 1] = 0.0
#        else:
#            z = self._layers[-1].get_signal()  # z = W * a + b
#            d = self._layers[-1].derivative(z)  # g'(z)
#            delta[n_layers - 1] = np.multiply(delta[n_layers], d)
#
#        for i in range(n_layers - 2, -1, -1):
#            z = self._layers[i].get_signal()  # z = W * a + b
#            d = self._layers[i].derivative(z)  # g'(z)
#            delta[i] = np.multiply(delta[i + 1], d)

        num_layers = len(self._layers)
        if num_layers > 0:
            first_layer = self._layers[0]
        else:
            first_layer = self._output

        first_layer._backward(y)  # First layer computes backward step


class BaseLayer(with_metaclass(abc.ABCMeta)):
    """This is the base class for all layers.
    """
    def __init__(self, num_nodes=None, nodes=None, weights=None):

        self._num_nodes = num_nodes
        self._nodes = nodes
        self.set_weights(weights)

        self._all_same = True
        if isinstance(nodes, list):
            self._all_same = False

        self._prev_layer = None
        self._next_layer = None

    def reset(self):

        self._signal = None
        self._activation = None
        self._delta = None
        self._grad = None

    def get_num_nodes(self):

        return self._num_nodes

    def connect_next(self, layer):

        self._next_layer = layer

    def connect_prev(self, layer):

        self._prev_layer = layer

        if layer is None:
            self._weights = None
        elif self._weights is None:
            self._weights = _init_weights(self.get_num_nodes(),
                                          layer.get_num_nodes())

    def get_weights(self):

        return self._weights

    def set_weights(self, weights):

        if weights is not None:
            self._weights = np.asarray(weights)
        else:
            self._weights = weights

    def get_signal(self):

        return self._signal

    def get_activation(self):

        return self._activation

    def get_derivative(self, z=None):

        if z is not None:
            if self._all_same:
                self._derivative = self._nodes.f(z)
            else:
                self._derivative = np.zeros((self._num_nodes, z.shape[1]))
                for i in range(self._num_nodes):
                    self._derivative[i, :] = self._nodes[i].f(z[i, :])

        return self._derivative

    def get_delta(self):

        return self._delta

    def get_grad(self):

        return self._grad

    def _forward(self, X):

        a = self._prev_layer._forward(X)
        self._signal = np.dot(self._weights, a)

        if self._all_same:
            self._activation = self._nodes.f(self._signal)
        else:
            self._activation = np.zeros((self._num_nodes, 1))
            for i in range(self._num_nodes):
                self._activation[i] = self._nodes[i].f(self._signal[i])

        return self._activation

    def _backward(self, y):

        # Compute delta in above layers
        delta2 = self._next_layer._backward(y)

        # Compute delta in this layer
        z = self.get_signal()
        d = self.get_derivative(z)
        W = self.get_weights()
        self._delta = np.multiply(np.dot(W.T, delta2), d)

        # Compute gradient
        aj = self._prev_layer.get_activation()  # Activation of previous layer
        self._grad = np.dot(self._delta, aj.T)

        return self._delta


class InputLayer(BaseLayer):
    """Represents an input layer.
    """
    def __init__(self, num_nodes=None):

        super(InputLayer, self).__init__(num_nodes=num_nodes,
                                         nodes=IdentityNode(),
                                         weights=None)

    def connect_prev(self, layer):

        raise ValueError("Cannot add a previous layer to the input layer!")

    def get_signal(self):

        raise ValueError("The input layer doesn't have an input signal!")

    def _forward(self, X):

        self._activation = X

        return self._activation

    def _backward(self):

        raise ValueError("The input layer doesn't depend on weights!")


class HiddenLayer(BaseLayer):
    """Represents a hidden layer.
    """
    pass


class OutputLayer(BaseLayer):
    """Represents the output layer.
    """
    def __init__(self, loss, num_nodes=None, nodes=None, weights=None):

        super(OutputLayer, self).__init__(num_nodes=num_nodes,
                                          nodes=nodes,
                                          weights=weights)

        self.set_loss(loss)

    def connect_next(self, layer):

        raise ValueError("Cannot add a next layer to the output layer!")

    def get_grad(self):
        """The gradient of the composition of the loss and output layer.
        """
        return super(OutputLayer, self).get_grad()

    def _backward(self, y):

        # Compute delta in the output layer
        z = self.get_signal()  # z = Wi * aj
        d1 = self.get_derivative(z)  # g'(z) = g'(Wi * aj)
        d2 = self.get_loss().derivative(y)  # dL / dz

        self._delta = np.multiply(d1, d2)

        # Compute gradient
        aj = self._prev_layer.get_activation()  # Activation of previous layer
        self._grad = np.dot(self._delta, aj.T)

        return self._delta

    def get_loss(self):

        return self._loss

    def set_loss(self, loss):

        self._loss = loss


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
        if isinstance(x, np.ndarray):
            return np.reciprocal(1.0 + np.exp(-x))
        else:
            return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, x):
        f = self.f(x)
        if isinstance(x, np.ndarray):
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

        self.target = target

    def get_target(self):

        return self.target


class SquaredSumLoss(BaseLoss):
    """A squared error loss function

        f(x) = (1 / 2) * \sum_{i=1}^n (x_i - t_i)².
    """
    def f(self, x):

        return 0.5 * np.sum((x - self.target) ** 2.0)

    def derivative(self, x):

        return (x - self.target)


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
