# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.bases` module includes several base classes
for using and creating algorithms.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Thu Feb 20 17:42:16 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from six import with_metaclass
import abc
import functools

import parsimony.utils.consts as consts
import parsimony.functions.properties as properties

__all__ = ["BaseAlgorithm", "check_compatibility",
           "ImplicitAlgorithm", "ExplicitAlgorithm",
           "IterativeAlgorithm", "InformationAlgorithm", "KernelAlgorithm"]


class BaseAlgorithm(with_metaclass(abc.ABCMeta, object)):

    @staticmethod
    def check_compatibility(function, required_properties):
        """Check if the function considered implements the given properties.
        """
        if not isinstance(function, (list, tuple)):
            function = [function]

        for f in function:
            for prop in required_properties:
                if isinstance(prop, properties.OR):
                    if not prop.evaluate(f):
                        raise ValueError("%s does not implement all "
                                         "properties %s" % (str(f), str(prop)))
                elif not isinstance(f, prop):
                    raise ValueError("%s does not implement interface %s" %
                                     (str(f), str(prop)))

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def get_params(self):
        raise NotImplementedError('Method "get_params" has not been '
                                  'implemented.')

    def reset(self):
        """Resets the algorithm so that it is as if just created.

        Override in order to reset more things, but remember to call the base
        class' reset() method.
        """
        # TODO: Keep this list up to date!
        if isinstance(self, IterativeAlgorithm):
            self.iter_reset()
        if isinstance(self, InformationAlgorithm):
            self.info_reset()


# TODO: Replace the one in BaseAlgorithm.
def check_compatibility(f):
    """Automatically checks if a function implements a given set of properties.
    """
    @functools.wraps(f)
    def wrapper(self, function, *args, **kwargs):

        BaseAlgorithm.check_compatibility(function, self.INTERFACES)

        return f(self, function, *args, **kwargs)

    return wrapper


def force_reset(f):
    """Decorate run with this method to force a reset of your algorithm.

    Automatically resets an algorithm by checking the implementing
    classes and calling the appropriate reset methods.
    """
    @functools.wraps(f)
    def wrapper(self, function, *args, **kwargs):

        # Add more subclasses here if necessary.
        if isinstance(self, IterativeAlgorithm):
            self.iter_reset()
        if isinstance(self, InformationAlgorithm):
            self.info_reset()

        return f(self, function, *args, **kwargs)

    return wrapper


class ImplicitAlgorithm(with_metaclass(abc.ABCMeta, BaseAlgorithm)):
    """Implicit algorithms are algorithms that do not utilise a loss function.
    Implicit algorithms instead minimise or maximise some underlying function
    implicitly, usually from the data.

    Parameters
    ----------
    X : One or more data matrices.
    """
    @abc.abstractmethod
    def run(X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be '
                                  'specialised!')


class ExplicitAlgorithm(with_metaclass(abc.ABCMeta, BaseAlgorithm)):
    """Explicit algorithms are algorithms that minimises a given function.
    The function is explicitly minimised from properties of said function.

    Implementing classes should update the INTERFACES class variable with
    the properties that function must implement. Defaults to a list with one
    element, the Function.
    """
    INTERFACES = [properties.Function]

    @abc.abstractmethod
    def run(function, x, **kwargs):
        """This function obtains a minimiser of a give function.

        Parameters
        ----------
        function : The function to minimise.

        x : A starting point.
        """
        raise NotImplementedError('Abstract method "run" must be '
                                  'specialised!')


class IterativeAlgorithm(object):
    """Algorithms that require iterative steps to achieve the goal.

    Fields
    ------
    max_iter : Non-negative integer. The maximum number of allowed iterations.

    min_iter : Non-negative integer less than or equal to max_iter. The minimum
            number of iterations that must be performed. Default is 1.

    num_iter : Non-negative integer greater than or equal to min_iter. The
            number of iterations performed by the iterative algorithm. All
            algorithms that inherit from IterativeAlgortihm MUST call
            iter_reset before every run.

    Parameters
    ----------
    max_iter : int
        A non-negative integer. The maximum number of allowed iterations.

    min_iter : int
        A non-negative integer. The minimum number of required iterations.

    callback : Callable
        A callable that accepts a dictionary with parameters and their values.
        Usually callback will be called with the output of locals() at each
        iteration of the algorithm.
    """
    def __init__(self, max_iter=consts.MAX_ITER, min_iter=1, callback=None,
                 **kwargs):

        super(IterativeAlgorithm, self).__init__(**kwargs)

        self.max_iter = max_iter
        self.min_iter = min_iter
        self.callback = callback
        self.num_iter = 0

        self.iter_reset()

    def iter_reset(self):

        self.num_iter = 0


class InformationAlgorithm(object):
    """Algorithms that produce information about their run.

    Implementing classes should update the INFO_PROVIDED class variable with
    the information provided by the algorithm. Defauls to an empty list.

    ALL algorithms that inherit from InformationAlgorithm MUST add force_reset
    as a decorator to the run method.

    Fields
    ------
    info_ret : Dictionary. The algorithm outputs are collected in this
            dictionary.

    info : List of utils.Info. The identifiers for the requested information
            outputs. The algorithms will store the requested outputs in
            self.info.

    INFO_PROVIDED : List of utils.Info. The allowed output identifiers. The
            implementing class should update this list with the
            provided/allowed outputs.

    Examples
    --------
    >>> import parsimony.algorithms as algorithms
    >>> from parsimony.algorithms.utils import Info
    >>> from parsimony.functions.losses import LinearRegression
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> gd = algorithms.gradient.GradientDescent(info=[Info.fvalue])
    >>> gd.info_copy()
    ['fvalue']
    >>> lr = LinearRegression(X=np.random.rand(10,15), y=np.random.rand(10,1))
    >>> beta = gd.run(lr, np.random.rand(15, 1))
    >>> fvalue = gd.info_get(Info.fvalue)
    >>> round(fvalue[0], 10)
    0.068510926
    >>> round(fvalue[-1], 13)
    1.9e-12
    """
    INFO_PROVIDED = []

    def __init__(self, info=[], **kwargs):
        """
        Parameters
        ----------
        info : List or tuple of utils.Info. The identifiers for the run
                information to return.
        """
        super(InformationAlgorithm, self).__init__(**kwargs)

        if not isinstance(info, (list, tuple)):
            self.info = [info]
        else:
            self.info = list(info)
        self.info_ret = dict()

        self.check_info_compatibility(self.info)

    def info_get(self, nfo=None):
        """Returns the computed information about the algorithm run.

        Parameters
        ----------
        nfo : utils.Info. The identifier to return information about. If nfo is
                None, all information is returned in a dictionary.
        """
        if nfo is None:
            return self.info_ret
        else:
            return self.info_ret[nfo]

    def info_set(self, nfo, value):
        """Sets the computed information about the algorithm run identified by
        nfo.

        Parameters
        ----------
        nfo : utils.Info. The identifier to for the computed information about.

        value : object. The value to associate with nfo.
        """
        self.info_ret[nfo] = value

    def info_provided(self, nfo):
        """Returns true if the current algorithm provides the given
        information, and False otherwise.
        """
        return nfo in self.INFO_PROVIDED

    def info_requested(self, nfo):
        """Returns true if the the given information was requested, and False
        otherwise.
        """
        return nfo in self.info

    def info_add_request(self, nfo):
        """Add a request to the algorithm's list of requested info.
        """
        return self.info.append(nfo)

    def info_reset(self):
        """Resets the information saved in the previous run. The info_ret
        field, a dictionary, is cleared.
        """
        self.info_ret.clear()

    def info_copy(self):
        """Returns a shallow copy of the requested information.
        """
        return list(self.info)

    def check_info_compatibility(self, info):
        """Check if the requested information is provided.

        Parameters
        ----------
        info : A list of utils.Info. The identifiers for information that
                should be computed.
        """
        for i in info:
            if not self.info_provided(i):
                raise ValueError("Requested information (%s) not provided."
                                 % (str(i),))


class KernelAlgorithm(object):
    """Algorithms that use Mercer kernels.

    Implementing classes should have a field kernel and supply a get_kernel
    method.

    Fields
    ------
    kernel_get : kernel object, optional
        Returns the kernel. Default is a linear kernel.

    Examples
    --------
    >>> import parsimony.algorithms.algorithms as algorithms
    >>> import parsimony.algorithms.utils as utils
    >>>
    >>> K = utils.LinearKernel()
    >>> smo = algorithms.SequentialMinimalOptimization(1.0, kernel=K)
    >>> # smo.kernel_get()
    """
    def __init__(self, kernel=None, **kwargs):
        """
        Parameters
        ----------
        kernel : kernel object, optional
            The kernel to use. Default is a linear kernel.
        """
        super(KernelAlgorithm, self).__init__(**kwargs)

        self.kernel = kernel

    def kernel_get(self):
        """Returns the kernel.
        """
        return self.kernel


if __name__ == "__main__":
    import doctest
    doctest.testmod()
