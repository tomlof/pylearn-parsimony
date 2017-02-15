# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.neural` module contains classes for neural
networks.

Loss functions should be stateless. Loss functions may be shared and copied
and should therefore not hold anything that cannot be recomputed the next time
it is called.

Created on Wed Feb 15 11:12:33 2017

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


class BaseNetwork(with_metaclass(abc.ABCMeta,
                                 properties.Function,
                                 properties.Gradient)):
    """This is the base class for all neural networks.
    """
    def __init__(self, input_size):

        self._layers = [InputLayer(input_size)]

    def add_layer(self, layer):

        self._layers[-1].connect_next(layer)  # Connect last layer to this
        layer.connect_prev(self._layers[-1])  # Connect this layer to last

        self._layers.append(layer)

#        return len(self._layers) - 1  # Return index of the layer

    def f(self, x):
        pass  # Implement!!


class BaseLayer(with_metaclass(abc.ABCMeta), object):
    """This is the base class for all layers.
    """
    def __init__(self, num_input_nodes=1, num_output_nodes=1, nodes=None):

        self._nodes = nodes
        self._num_input_nodes = num_input_nodes
        self._num_output_nodes = num_output_nodes

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

        if self._all_same:
            outputs = self._nodes.f(inputs)
        else:
            outputs = np.zeros((self.num_nodes, 1))
            for i in range(self.num_nodes):
                outputs[i] = self._nodes[i].f(inputs[i])

        return outputs

    def backward(self, grads):

        pass  # TODO: implement!!


class InputLayer(BaseLayer):
    """Represents an input layer.
    """
    def __init__(self, num_output_nodes=1):

        super(BaseLayer, self).__init__(num_input_nodes=0,
                                        num_output_nodes=num_output_nodes,
                                        IdentityNode())

    def connect_prev(self, layer):

        raise ValueError("Cannot add a previous layer to the input layer!")


class HiddenLayer(BaseLayer):
    """Represents a hidden layer.
    """
    def __init__(self, num_input_nodes=1, num_output_nodes=1, nodes=None):

        super(BaseLayer, self).__init__(num_input_nodes=num_input_nodes,
                                        num_output_nodes=num_output_nodes,
                                        nodes)


class OutputLayer(HiddenLayer):
    """Represents an output layer.
    """
    pass


class BaseNode(with_metaclass(abc.ABCMeta),
               properties.Function,
               properties.Gradient):

    pass


class IdentityNode(BaseNode):

    def f(self, x):
        return x

    def grad(self, x):
        if isinstance(np.ndarray):
            return np.ones(x.shape)
        else:
            return 1.0


class LogisticNode(BaseNode):

    def f(self, x):
        if isinstance(np.ndarray):
            return np.reciprocal(1.0 + np.exp(-x))
        else:
            return 1.0 / (1.0 + np.exp(-x))

    def grad(self, a):
        f = self.f(a)
        if isinstance(np.ndarray):
            return np.multiply(f, 1.0 - f)
        else:
            return f * (1.0 - f)
