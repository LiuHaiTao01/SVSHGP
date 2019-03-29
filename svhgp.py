# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import tensorflow as tf

from .. import kullback_leiblers, features
from .. import settings
from .. import transforms
from ..conditionals import conditional, Kuu
from ..decors import params_as_tensors, autoflow
from ..models.model import GPModel
from ..params import DataHolder
from ..params import Minibatch
from ..params import Parameter
from ..likelihoods import HeteroGaussian


class SVHGP(GPModel):
    """
    This is the Sparse Variational Heteroscedastic GP (SVHGP) 
    implemented by Haitao Liu.
    htliu@ntu.edu.sg
    """

    def __init__(self, X, Y, kern, kern_g, likelihood, feat=None, feat_g=None,
                 mean_function=None,
                 mu0_g=0., # constant mean function for g
                 num_latent=None,
                 q_diag=False,
                 q_diag_g=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 Z_g=None,
                 num_data=None,
                 q_mu=None,
                 q_sqrt=None,
                 q_mu_g=None,
                 q_sqrt_g=None,
                 **kwargs):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x P
        - kern, likelihood, mean_function are appropriate GPflow objects for latent function f
        - kern_g is the kernel for latent log noise function g
        - Z is a matrix of pseudo inputs, size M x D, for f
        - Z_g is a matrix of pseudo inputs, size M' x D, for g
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag (q_diag_g) is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        assert isinstance(self.likelihood, HeteroGaussian), 'likelihood should be HeteroGaussian'
        self.kern_g = kern_g
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.q_diag_g, self.whiten = q_diag, q_diag_g, whiten
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.feature_g = features.inducingpoint_wrapper(feat_g, Z_g)
        self.mu0_g = Parameter(mu0_g, dtype=settings.float_type)

        # init variational parameters
        num_inducing = len(self.feature)
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)
        num_inducing_g = len(self.feature_g)
        self._init_variational_parameters_g(num_inducing_g, q_mu_g, q_sqrt_g, q_diag_g)


    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=settings.float_type)  # M x P

        if q_sqrt is None:
            if self.q_diag:
                self.q_sqrt = Parameter(np.ones((num_inducing, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt = np.array([np.eye(num_inducing, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # P x M x M
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(num_inducing, self.num_latent))  # L/P x M x M

    def _init_variational_parameters_g(self, num_inducing_g, q_mu_g, q_sqrt_g, q_diag_g):
        q_mu_g = np.zeros((num_inducing_g, self.num_latent)) if q_mu_g is None else q_mu_g
        self.q_mu_g = Parameter(q_mu_g, dtype=settings.float_type)  # M x P

        if q_sqrt_g is None:
            if self.q_diag_g:
                self.q_sqrt_g = Parameter(np.ones((num_inducing_g, self.num_latent), dtype=settings.float_type),
                                        transform=transforms.positive)  # M x P
            else:
                q_sqrt_g = np.array([np.eye(num_inducing_g, dtype=settings.float_type) for _ in range(self.num_latent)])
                self.q_sqrt_g = Parameter(q_sqrt_g, transform=transforms.LowerTriangular(num_inducing_g, self.num_latent))  # P x M x M
        else:
            if q_diag_g:
                assert q_sqrt_g.ndim == 2
                self.num_latent = q_sqrt_g.shape[1]
                self.q_sqrt_g = Parameter(q_sqrt_g, transform=transforms.positive)  # M x L/P
            else:
                assert q_sqrt_g.ndim == 3
                self.num_latent = q_sqrt_g.shape[0]
                num_inducing_g = q_sqrt_g.shape[1]
                self.q_sqrt_g = Parameter(q_sqrt_g, transform=transforms.LowerTriangular(num_inducing_g, self.num_latent))  # L/P x M x M

    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            K = None
        else:
            K = Kuu(self.feature, self.kern, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)

    @params_as_tensors
    def build_prior_KL_g(self):
        # KL divergence between q(g; mu_g, sigma_g) and p(g; mu0_g, Kuu)
        if self.whiten:
            K_g = None
        else:
            K_g = Kuu(self.feature_g, self.kern_g, jitter=settings.numerics.jitter_level)  # (P x) x M x M

        q_mu_g_mu0 = self.mu0_g - self.q_mu_g
        return kullback_leiblers.gauss_kl(q_mu_g_mu0, self.q_sqrt_g, K_g)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get prior KL_g.
        KL_g = self.build_prior_KL_g()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False, full_output_cov=False)

        # Get conditionals for log noise g
        fmean_g, fvar_g = self._build_predict_g(self.X, full_cov=False, full_output_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, fmean_g, fvar_g, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.float_type) / tf.cast(tf.shape(self.X)[0], settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL - KL_g

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False, full_output_cov=False):
        mu, var = conditional(Xnew, self.feature, self.kern, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mean_function(Xnew), var

    @params_as_tensors
    def _build_predict_g(self, Xnew, full_cov=False, full_output_cov=False):
        q_mu_g_mu0 = self.q_mu_g - self.mu0_g
        mu, var = conditional(Xnew, self.feature_g, self.kern_g, q_mu_g_mu0, q_sqrt=self.q_sqrt_g, full_cov=full_cov,
                              white=self.whiten, full_output_cov=full_output_cov)
        return mu + self.mu0_g, var

    @autoflow((settings.float_type, [None, None]))
    def predict_g(self, Xnew):
        return self._build_predict_g(Xnew)

    @autoflow((settings.float_type, [None, None]))
    def predict_y(self, Xnew):
        pred_f_mean, pred_f_var = self._build_predict(Xnew)
        pred_g_mean, pred_g_var = self._build_predict_g(Xnew)
        return self.likelihood.predict_mean_and_var(pred_f_mean, pred_f_var, pred_g_mean, pred_g_var)