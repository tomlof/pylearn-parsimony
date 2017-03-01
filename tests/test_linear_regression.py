# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:21:23 2014

Copyright (c) 2013-2017, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from nose.tools import assert_less, assert_almost_equal

import numpy as np

import parsimony.algorithms.proximal as proximal
import parsimony.utils.consts as consts
try:
    from .tests import TestCase  # When imported as a package.
except (ValueError, SystemError):
    from tests import TestCase  # When run as a program.


class TestLinearRegression(TestCase):

    def test_overdetermined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        n, p = 100, 50

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector(p)

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 150
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        mse = np.linalg.norm(beta_parsimony - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        if abs(f_star) > consts.TOLERANCE:
            err = abs(f_parsimony - f_star) / f_star
        else:
            err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give "
                               "the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                         algorithm_params=dict(max_iter=max_iter),
                                         mean=False)
        lr.fit(X, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-4, "The found regression vector is not correct.")

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-7, "The found regression vector does not give "
                               "the correct function value.")

    def test_underdetermined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector(p)

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 300
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-1, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give "
                               "the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                         algorithm_params=dict(max_iter=max_iter),
                                          mean=False)
        lr.fit(X, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-1, "The found regression vector is not correct.")

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_determined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        n, p = 50, 50

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector(p)

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 7000
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                         algorithm_params=dict(max_iter=max_iter),
                                         mean=False)
        lr.fit(X, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

    def test_intercept1(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.gradient as gradient
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=False)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p - 1, p - 1) \
            + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        X0 = np.random.multivariate_normal(mean, Sigma, n)
        X_parsimony = np.hstack((np.ones((n, 1)), X0))
        X_spams = np.hstack((X0, np.ones((n, 1))))

        beta_star = start_vector.get_vector(p)

        e = 0.01 * np.random.rand(n, 1)
        y = np.dot(X_parsimony, beta_star) + e

        eps = 1e-8
        max_iter = 1000
        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X_parsimony, y, mean=True)
        beta_start = start_vector.get_vector(p)

        beta_parsimony = gd.run(linear_regression, beta_start)

        try:
            import spams

            params = {"loss": "square",
                      "regul": "l2",
                      "lambda1": 0.0,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": True}

            beta_spams, optim_info = \
                spams.fistaFlat(Y=np.asfortranarray(y),
                                X=np.asfortranarray(X_spams),
                                W0=np.asfortranarray(beta_start),
                                return_optim_info=True,
                                **params)

#            print beta_spams

        except ImportError:
            beta_spams = np.asarray(
                    [[0.09736768], [0.69854941], [0.48793715], [0.75698677],
                     [0.44423199], [0.63262493], [0.30397824], [0.5815484],
                     [0.02765551], [0.63991869], [1.09671465], [0.21529914],
                     [0.39480577], [0.54851789], [-0.02412213], [0.69405293],
                     [0.18821586], [0.89556287], [-0.14179676], [0.86230118],
                     [0.46760193], [0.62187006], [0.21463264], [0.40930455],
                     [1.00249679], [1.08387997], [0.1115664], [0.66241187],
                     [0.42912028], [0.0020646], [1.087087], [0.84161254],
                     [0.37992596], [0.45365878], [0.52357722], [0.0683171],
                     [0.79540688], [0.65421616], [0.47634391], [0.24049712],
                     [0.55384988], [0.33100698], [0.12267719], [0.62996432],
                     [0.88073445], [0.70778668], [0.83411801], [0.83597934],
                     [0.14499151], [0.82389504], [0.78075443], [0.99507837],
                     [0.47733893], [0.56237854], [0.7443968], [0.30659158],
                     [0.29745792], [0.74846412], [0.06055889], [-0.17837795],
                     [0.50033042], [0.62078627], [0.58955146], [-0.04940678],
                     [1.16326298], [0.33954734], [0.74985271], [0.05866496],
                     [0.70692727], [0.29520974], [0.09383234], [0.39026236],
                     [0.65104699], [0.83970364], [0.6179541], [0.60286729],
                     [0.47713691], [0.975948], [0.78007776], [0.33100383],
                     [0.41049941], [0.6689349], [0.64447833], [0.11711684],
                     [0.44066274], [0.82042655], [0.48411089], [0.28608364],
                     [0.13726529], [0.76605214]])


        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert_less(berr, 2.02, "The found regression vector is not correct.")

        berr = np.linalg.norm(beta_spams - beta_star)
#        print "berr:", berr
        assert_less(berr, 4.3, "The found regression vector is not correct.")

        f_star = linear_regression.f(beta_star)

        f_parsimony = linear_regression.f(beta_parsimony)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert_less(ferr, 5e-5, msg="The found regression vector does not "
                                    "give the correct function value.")

        beta_spams = np.vstack((beta_spams[p - 1, :],
                                beta_spams[0:p - 1, :]))
        f_spams = linear_regression.f(beta_spams)
        ferr = abs(f_spams - f_star)
#        print "ferr:", ferr
        assert_less(ferr, 5e-4, msg="The found regression vector does not "
                                    "give the correct function value.")

        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                         algorithm_params=dict(max_iter=max_iter),
                                         mean=True)
        lr.fit(X_parsimony, y, beta_start)

        berr = np.linalg.norm(lr.beta - beta_star)
#        print "berr:", berr
        assert_almost_equal(berr, 2.010556,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_est = linear_regression.f(lr.beta)
        f_star = linear_regression.f(beta_star)
        err = abs(f_est - f_star)
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

#    def test_linear_regression_intercept2(self):
#
#        from parsimony.functions.combinedfunctions import CombinedFunction
#        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.penalties import L2
#        import parsimony.algorithms.gradient as gradient
#        import parsimony.utils.start_vectors as start_vectors
#
#        np.random.seed(42)
#
#        start_vector = start_vectors.RandomStartVector(normalise=False)
#
#        n, p = 60, 90
#
#        alpha = 1.0
#        Sigma = alpha * np.eye(p - 1, p - 1) \
#              + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
#        mean = np.zeros(p - 1)
#        X0 = np.random.multivariate_normal(mean, Sigma, n)
#        X_parsimony = np.hstack((np.ones((n, 1)), X0))
#        X_spams = np.hstack((X0, np.ones((n, 1))))
#
#        beta_star = start_vector.get_vector(p)
#
#        e = 0.01 * np.random.rand(n, 1)
#        y = np.dot(X_parsimony, beta_star) + e
#
#        eps = 1e-8
#        max_iter = 2500
#
#        k = 0.318
#        function = CombinedFunction()
#        function.add_function(LinearRegression(X_parsimony, y, mean=True))
#        function.add_penalty(L2Squared(k, penalty_start=1))
#
#        gd = gradient.GradientDescent(eps=eps, max_iter=max_iter)
#        beta_start = start_vector.get_vector(p)
#        beta_parsimony = gd.run(function, beta_start)
#
#        try:
#            import spams
#
#            params = {"loss": "square",
#                      "regul": "l2",
#                      "lambda1": k,
#                      "max_it": max_iter,
#                      "tol": eps,
#                      "ista": True,
#                      "numThreads": -1,
#                      "intercept": True,
#                     }
#
#            beta_start = np.vstack((beta_start[1:p, :],
#                                    beta_start[0, :]))
#            beta_spams, optim_info = \
#                    spams.fistaFlat(Y=np.asfortranarray(y),
#                                    X=np.asfortranarray(X_spams),
#                                    W0=np.asfortranarray(beta_start),
#                                    return_optim_info=True,
#                                    **params)
#
##            print beta_spams
#
#        except ImportError:
#            beta_spams = np.asarray(
#                    )
#
#        beta_spams = np.vstack((beta_spams[p - 1, :],
#                                beta_spams[0:p - 1, :]))
#        mse = np.linalg.norm(beta_parsimony - beta_spams) #\
#                #/ np.linalg.norm(beta_spams)
#        print "mse:", mse
##        assert_almost_equal(mse, 0.367913,
##                           msg="The found regression vector is not correct.",
##                           places=5)
#        print np.hstack((beta_star, beta_parsimony, beta_spams))
#
#        f_parsimony = function.f(beta_parsimony)
#        f_spams = function.f(beta_spams)
##        if abs(f_star) > consts.TOLERANCE:
##            err = abs(f_parsimony - f_star) / f_star
##        else:
#        err = abs(f_parsimony - f_spams) #/ f_spams
#        print "err:", err
##        assert_less(err, 5e-05, msg="The found regression vector does not " \
##                                    "give the correct function value.")

    def test_l1(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 5100
        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        l1 = L1(l=l)
        function = CombinedFunction()
        function.add_loss(linear_regression)
        function.add_prox(l1)

        beta_start = start_vector.get_vector(p)

        beta_parsimony = fista.run(function, beta_start)

        mu = consts.TOLERANCE
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=proximal.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        reg_est.fit(X, y)

#        rreg_est = estimators.RidgeRegression_L1_TV(
#                    k=k, l=l, g=g,
#                    A=A, mu=mu,
#                    algorithm=explicit.FISTA(),
#                    algorithm_params=dict(eps=eps, max_iter=max_iter),
#                    mean=False)
#        rreg_est.fit(X, y)

        rreg_est_2 = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                       algorithm=proximal.FISTA(),
                                                       algorithm_params=dict(eps=eps,
                                                                             max_iter=max_iter),
                                                       mean=False)
        rreg_est_2.fit(X, y)

        lasso = estimators.Lasso(l,
                                 algorithm=proximal.FISTA(),
                                 algorithm_params=dict(eps=eps,
                                                       max_iter=max_iter),
                                 mean=False)
        lasso.fit(X, y)

        re = np.linalg.norm(beta_parsimony - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        re = np.linalg.norm(reg_est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

#        re = np.linalg.norm(rreg_est.beta - beta_star) \
#                / np.linalg.norm(beta_star)
##        print "re:", re
#        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(rreg_est_2.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)

        f_parsimony = function.f(beta_parsimony)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        f_est = function.f(reg_est.beta)
        err = abs(f_est - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

#        f_rest = function.f(rreg_est.beta)
#        err = abs(f_rest - f_star) / f_star
##        print "err:", err
#        assert_less(err, 5e-5, "The found regression vector does not give " \
#                               "the correct function value.")

        f_rest_2 = function.f(rreg_est_2.beta)
        err = abs(f_rest_2 - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        f_lasso = function.f(lasso.beta)
        err = abs(f_lasso - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_l1_intercept(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90 + 1

        alpha = 0.9
        V = np.random.randn(p - 1, p - 1)
        Sigma = alpha * np.eye(p - 1, p - 1) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p - 1)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr,
                                        intercept=True)

        eps = 1e-8
        max_iter = 3800
        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_prox(L1(l=l, penalty_start=1))

        beta_start = start_vector.get_vector(p)

        beta_parsimony = fista.run(function, beta_start)

        mu = consts.TOLERANCE
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=proximal.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        reg_est.fit(X, y)

#        rreg_est = estimators.RidgeRegression_L1_TV(
#                    k=k, l=l, g=g,
#                    A=A, mu=mu,
#                    algorithm=explicit.FISTA(eps=eps, max_iter=max_iter),
#                    penalty_start=1,
#                    mean=False)
#        rreg_est.fit(X, y)

        rreg_est_2 = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                       algorithm=proximal.FISTA(),
                                                       algorithm_params=dict(eps=eps,
                                                                             max_iter=max_iter),
                                                       penalty_start=1,
                                                       mean=False)
        rreg_est_2.fit(X, y)

        lasso = estimators.Lasso(l,
                                 algorithm=proximal.FISTA(),
                                 algorithm_params=dict(eps=eps,
                                                       max_iter=max_iter),
                                 penalty_start=1,
                                 mean=False)
        lasso.fit(X, y)

        re = np.linalg.norm(beta_parsimony - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(reg_est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

#        re = np.linalg.norm(rreg_est.beta - beta_star) \
#                / np.linalg.norm(beta_star)
##        print "re:", re
#        assert_less(re, 5e-3, "The found regression vector is not correct.")

        re = np.linalg.norm(rreg_est_2.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)

        f_parsimony = function.f(beta_parsimony)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        f_est = function.f(reg_est.beta)
        err = abs(f_est - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

#        f_rest = function.f(rreg_est.beta)
#        err = abs(f_rest - f_star) / f_star
##        print "err:", err
#        assert_less(err, 5e-5, "The found regression vector does not give " \
#                               "the correct function value.")

        f_rest_2 = function.f(rreg_est_2.beta)
        err = abs(f_rest_2 - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        f_lasso = function.f(lasso.beta)
        err = abs(f_lasso - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give "
                               "the correct function value.")

    def test_l2(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L2Squared
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 7500

        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector(p)

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(L2Squared(k))
        beta_penalty = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give "
                               "the correct function value.")

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_prox(L2Squared(k))
        beta_prox = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_prox - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_prox = function.f(beta_prox)
        err = abs(f_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give "
                               "the correct function value.")

        function = CombinedFunction()
        function.add_loss(RidgeRegression(X, y, k, mean=False))
        beta_rr = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_l2_intercept(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L2Squared
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        V = np.random.randn(p - 1, p - 1)
        Sigma = alpha * np.eye(p - 1, p - 1) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 0.0

        A = np.eye(p - 1)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr,
                                        intercept=True)

        eps = 1e-8
        max_iter = 1500

        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector(p)

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(L2Squared(k, penalty_start=1))
        beta_penalty = fista.run(function, beta_start)

        re = np.linalg.norm(beta_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_prox(L2Squared(k, penalty_start=1))
        beta_prox = fista.run(function, beta_start)

        re = np.linalg.norm(beta_prox - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_prox = function.f(beta_prox)
        err = abs(f_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        function = CombinedFunction()
        function.add_loss(RidgeRegression(X, y, k, penalty_start=1,
                                              mean=False))
        beta_rr = fista.run(function, beta_start)

        re = np.linalg.norm(beta_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        mu = consts.TOLERANCE
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=proximal.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        reg_est.fit(X, y)

        re = np.linalg.norm(reg_est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(reg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

#        rreg_est = estimators.RidgeRegression_L1_TV(
#                    k=k, l=l, g=g,
#                    A=A, mu=mu,
#                    algorithm=explicit.FISTA(eps=eps, max_iter=max_iter),
#                    penalty_start=1,
#                    mean=False)
#        rreg_est.fit(X, y)

#        re = np.linalg.norm(rreg_est.beta - beta_star) \
#                / np.linalg.norm(beta_star)
##        print "re:", re
#        assert_less(re, 5e-3, "The found regression vector is not correct.")

#        f_star = function.f(beta_star)
#        f_rr = function.f(rreg_est.beta)
#        err = abs(f_rr - f_star) / f_star
##        print "err:", err
#        assert_less(err, 5e-4, "The found regression vector does not give " \
#                               "the correct function value.")

        rreg_est_2 = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                       algorithm=proximal.FISTA(),
                                                       algorithm_params=dict(eps=eps,
                                                                             max_iter=max_iter),
                                                       penalty_start=1,
                                                       mean=False)
        rreg_est_2.fit(X, y)

        re = np.linalg.norm(rreg_est_2.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(rreg_est_2.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_tv(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        A = tv.linear_operator_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 5000

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth = function.f(beta_nonsmooth)
        err = abs(f_nonsmooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth = function.f(beta_smooth)
        err = abs(f_smooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

    def test_tv_intercept(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape) + 1

        alpha = 0.9
        Sigma = alpha * np.eye(p - 1, p - 1) \
            + (1.0 - alpha) * np.random.randn(p - 1, p - 1)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.001  # Cannot be zero.
        g = 1.618

        A = tv.linear_operator_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 8800

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr, intercept=True)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=1))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        re = np.linalg.norm(beta_nonsmooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth = function.f(beta_nonsmooth)
        err = abs(f_nonsmooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr,
                                          intercept=True)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=1))
            beta_smooth = fista.run(function, beta_smooth)

        re = np.linalg.norm(beta_smooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth = function.f(beta_smooth)
        err = abs(f_smooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 12500
        mu = mu_min
        reg_est = estimators.LinearRegressionL1L2TV(
                                      l, k, g, A, mu=mu,
                                      algorithm=proximal.StaticCONESTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      penalty_start=1,
                                      mean=False)
        reg_est.fit(X, y)

        re = np.linalg.norm(reg_est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(reg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        rreg_est = estimators.LinearRegressionL1L2TV(l, k, g, A=A, mu=mu,
                                                     algorithm=proximal.StaticCONESTA(),
                                                     algorithm_params=dict(eps=eps,
                                                                           max_iter=max_iter),
                                                     penalty_start=1,
                                                     mean=False)
        rreg_est.fit(X, y)

        re = np.linalg.norm(rreg_est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(rreg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

    def test_gl(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [list(range(0, 2 * int(p / 3))), list(range(int(p / 3), p))]
        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.001  # Cannot be zero.
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 7000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        berr = np.linalg.norm(beta_nonsmooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-3, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_smooth = fista.run(function, beta_smooth)

        berr = np.linalg.norm(beta_smooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "berr:", berr
        assert_less(berr, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-3, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 7000
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=max_iter),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_gl_intercept(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90 + 1
        groups = [list(range(1, 2 * int(p / 3))), list(range(int(p / 3), p))]
        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups, weights=weights,
                                           penalty_start=1)

        alpha = 0.9
        V = np.random.randn(p - 1, p - 1)
        Sigma = alpha * np.eye(p - 1, p - 1) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p - 1)
        M0 = np.random.multivariate_normal(mean, Sigma, n)
        M = np.hstack((np.ones((n, 1)), M0))
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0001  # Cannot be zero.
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 8000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr, intercept=True)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=1))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        re = np.linalg.norm(beta_nonsmooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr,
                                          intercept=True)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=1))
            beta_smooth = fista.run(function, beta_smooth)

        re = np.linalg.norm(beta_smooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-6, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 5000
        rreg_est = estimators.LinearRegressionL1L2GL(l, k, g,
                                                     A=A, mu=mu,
                                                     algorithm=proximal.StaticCONESTA(),
                                                     algorithm_params=dict(eps=eps,
                                                                           max_iter=max_iter),
                                                     penalty_start=1,
                                                     mean=False)
        rreg_est.fit(X, y)

        re = np.linalg.norm(rreg_est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(rreg_est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

    def test_l1_l2(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 400

        fista = proximal.FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector(p)

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(L2Squared(k))
        function.add_prox(L1(l))
        beta_penalty = fista.run(function, beta_start)

        re = np.linalg.norm(beta_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        function = CombinedFunction()
        function.add_loss(RidgeRegression(X, y, k, mean=False))
        function.add_prox(L1(l))
        beta_rr = fista.run(function, beta_start)

        re = np.linalg.norm(beta_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        lasso = estimators.ElasticNet(l,
                                      algorithm=proximal.FISTA(),
                                      algorithm_params=dict(eps=eps,
                                                            max_iter=max_iter),
                                      mean=False)
        lasso.fit(X, y)

        f_star = function.f(beta_star)
        f_lasso = function.f(lasso.beta)
        err = abs(f_lasso - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

    def test_l1_tv(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 1.618

        A = tv.linear_operator_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 1800

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        function = None
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = \
                fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 2700
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_penalty = \
                fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_l1_gl(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [list(range(0, 2 * int(p / 3))), list(range(int(p / 3), p))]
        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.001  # Cannot be zero.
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 6000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 5200
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=max_iter),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_l2_tv(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 1.618

        A = tv.linear_operator_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 1900

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_nonsmooth_penalty = \
                fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        beta_nonsmooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_nonsmooth_prox = fista.run(function, beta_nonsmooth_prox)

        mse = np.linalg.norm(beta_nonsmooth_prox - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_prox = function.f(beta_nonsmooth_prox)
        err = abs(f_nonsmooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_smooth_penalty = \
                fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        beta_smooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_smooth_prox = fista.run(function, beta_smooth_prox)

        mse = np.linalg.norm(beta_smooth_prox - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_prox = function.f(beta_smooth_prox)
        err = abs(f_smooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-5, "The found regression vector does not give "
                               "the correct function value.")

    def test_l2_gl(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [list(range(0, 2 * int(p / 3))), list(range(int(p / 3), p))]
        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 3000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_nonsmooth_penalty = fista.run(function,
                                               beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_nonsmooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_nonsmooth_prox = fista.run(function, beta_nonsmooth_prox)

        mse = np.linalg.norm(beta_nonsmooth_prox - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_prox = function.f(beta_nonsmooth_prox)
        err = abs(f_nonsmooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            beta_smooth_penalty = fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_smooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L2Squared(k))
            beta_smooth_prox = fista.run(function, beta_smooth_prox)

        mse = np.linalg.norm(beta_smooth_prox - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_prox = function.f(beta_smooth_prox)
        err = abs(f_smooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 2500
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=max_iter),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_l1_l2_tv(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 1.618

        A = tv.linear_operator_from_shape(shape)
        snr = 100.0
        eps = 1e-8
        max_iter = 1200

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        function = None
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = \
                fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_smooth_penalty = \
                fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_l1_l2_gl(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2Squared
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        import parsimony.estimators as estimators

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [list(range(0, 2 * int(p / 3))), list(range(int(p / 3), p))]
        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 4000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector(p)

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = fista.run(function,
                                               beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2Squared(k))
            function.add_prox(L1(l))
            beta_smooth_penalty = fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_loss(RidgeRegression(X, y, k, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
            / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        max_iter = 2200
        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=mu,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=max_iter),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_estimators(self):

        import numpy as np
        import parsimony.estimators as estimators
        import parsimony.algorithms.gradient as gradient
        import parsimony.functions.nesterov.tv as tv
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulate.l1_l2_gl as l1_l2_gl
#        import parsimony.datasets.simulate.l1_l2_glmu as l1_l2_glmu
        import parsimony.utils.start_vectors as start_vectors
        from parsimony.functions import CombinedFunction
        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1, L2Squared

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        shape = (4, 4, 4)
        A = tv.linear_operator_from_shape(shape)

        n, p = 64, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.299125,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 0.969570,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        n, p = 100, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.154561,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        n, p = 100, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        beta = np.sort(beta, axis=0)
        beta[:10, :] = 0.0
        snr = 100.0

        l = 0.618  # L1 coefficient
        k = 1.0 - l  # Ridge coefficient
        g = 2.718  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        l = 0.0
        k = 0.0
        g = 0.0
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.019992,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        np.random.seed(42)
        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                         algorithm_params=dict(max_iter=10000))
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 2.963832,
                            msg="The found regression vector does not give "
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 0.0
        g = 0.0
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.070911,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 1.0 - 0.618
        g = 0.0
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.022994,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 0.0
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.029070,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 0.0
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 1.078159,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 0.0
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.032333,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 1.0 - 0.618
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.029266,
                            msg="The found regression vector does not give "
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.ISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 904.132727,
                            msg="The found regression vector does not give "
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        np.random.seed(42)
        lr = estimators.LinearRegressionL1L2TV(l, k, g, A,
                                               algorithm=proximal.FISTA(),
                                               algorithm_params=dict(max_iter=1000),
                                               mean=False)
        lr.fit(X, y)
        score = lr.score(X, y)
#        print "score:", score
        assert_almost_equal(score, 13.032530,
                            msg="The found regression vector does not give "
                                "the correct score value.",
                            places=5)

        # Test group lasso
        # ----------------
        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [list(range(0, 2 * int(p / 3))), list(range(int(p / 3), p))]
        weights = [1.5, 0.5]

        A = gl.linear_operator_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector(p)
        beta = np.sort(beta, axis=0)
        beta[:10, :] = 0.0

        snr = 20.0
        eps = 1e-8

        l = 0.0
        k = 0.001  # Cannot be zero.
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=17000,
                                                                      tau=0.9),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

        l = 0.618
        k = 0.001  # May not be zero.
        g = 1.618

        np.random.seed(42)

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=6500),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))
        function.add_prox(L1(l=l))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give "
                               "the correct function value.")

        np.random.seed(42)

        l = 0.0
        k = 0.618
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=2200),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))
        function.add_prox(L1(l=l))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-2, "The found regression vector does not give "
                               "the correct function value.")

        np.random.seed(42)

        l = 0.618
        k = 1.0 - l
        g = 1.618

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        est = estimators.LinearRegressionL1L2GL(l, k, g, A=A, mu=None,
                                                algorithm=proximal.StaticCONESTA(),
                                                algorithm_params=dict(eps=eps,
                                                                      max_iter=3900),
                                                penalty_start=0,
                                                mean=False)
        est.fit(X, y)

        re = np.linalg.norm(est.beta - beta_star) \
            / np.linalg.norm(beta_star)
#        print "re:", re
        assert_less(re, 5e-2, "The found regression vector is not correct.")

        function = CombinedFunction()
        function.add_loss(LinearRegression(X, y, mean=False))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A,
                                                  penalty_start=0))
        function.add_penalty(L2Squared(l=k))
        function.add_prox(L1(l=l))

        f_star = function.f(beta_star)
        f_rr = function.f(est.beta)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 5e-4, "The found regression vector does not give "
                               "the correct function value.")

    def test_large(self):

        import parsimony.algorithms.gradient as gradient
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        px = 10
        py = 10
        pz = 10
        shape = (pz, py, px)
        n, p = 100, np.prod(shape)

        A = tv.linear_operator_from_shape(shape)

        alpha = 0.9
        V = np.random.randn(p, p)
        Sigma = alpha * np.eye(p, p) \
            + (1.0 - alpha) * np.dot(V.T, V)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        beta = np.random.rand(p, 1) * 2.0 - 1.0
        beta = np.sort(beta, axis=0)
        beta[np.abs(beta) < 0.1] = 0.0

        y = np.dot(X, beta)

        eps = 1e-8
        max_iter = 10000

        k = 0.618
        l = 1.0 - k
        g = 1.618

        mu = None
        logreg_static = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                                          algorithm=proximal.StaticCONESTA(),
                                                          algorithm_params=dict(eps=eps,
                                                                                max_iter=max_iter),
                                                          mean=False)
        logreg_static.fit(X, y)
        err = logreg_static.score(X, y)
#        print err
        assert_less(err, 0.026,
                    msg="The found regression vector is not correct.")

        np.random.seed(42)
        mu = None
        logreg_dynamic = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                                           algorithm=proximal.CONESTA(),
                                                           algorithm_params=dict(eps=eps,
                                                                                 max_iter=max_iter),
                                                           mean=False)
        logreg_dynamic.fit(X, y)
        err = logreg_dynamic.score(X, y)
        assert_less(err, 0.02591,
                    msg="The found regression vector is not correct.")

        np.random.seed(42)
        mu = 5e-4
        logreg_fista = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                                         algorithm=proximal.FISTA(),
                                                         algorithm_params=dict(eps=eps,
                                                                               max_iter=max_iter),
                                                         mean=False)
        logreg_fista.fit(X, y)
        err = logreg_fista.score(X, y)
#        print err
        assert_less(err, 0.0259,
                    msg="The found regression vector is not correct.")

        np.random.seed(42)
        mu = 5e-4
        logreg_ista = estimators.LinearRegressionL1L2TV(l, k, g, A, mu=mu,
                                                        algorithm=proximal.ISTA(),
                                                        algorithm_params=dict(eps=eps,
                                                                              max_iter=max_iter),
                                                        mean=False)
        logreg_ista.fit(X, y)
        err = logreg_ista.score(X, y)
#        print err
        assert_less(err, 0.0350,
                    msg="The found regression vector is not correct.")

        np.random.seed(42)
        lr = estimators.LinearRegression(algorithm=gradient.GradientDescent(),
                                         algorithm_params=dict(eps=eps,
                                                               max_iter=max_iter),
                                         mean=False)
        lr.fit(X, y)
        err = lr.score(X, y)
#        print err
        assert_less(err, 5e-10,
                    msg="The found regression vector is not correct.")

if __name__ == "__main__":
    import unittest
    unittest.main()
