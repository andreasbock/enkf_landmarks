import torch
import math
import os
import scipy.linalg as la
import pickle

from lddmm import lddmm_forward, gauss_kernel
from ensemble import Ensemble
from utils import plot_landmarks

torch_dtype = torch.float32


class EnKFError(RuntimeError):
    pass


class EnsembleKalmanFilter:

    def __init__(self, ensemble_size, template, target, log_dir='./'):
        self.ensemble_size = ensemble_size
        self.template = template
        self.target = target
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.dim = template.shape[1]
        self.num_landmarks = template.shape[0]

        # Shooting/ODE parameters
        sigma = torch.tensor([1], dtype=torch_dtype)
        k = gauss_kernel(sigma=sigma)
        self.timesteps = 10
        self.shoot = lambda p0: lddmm_forward(p0, self.template, k, self.timesteps)[-1][1]

        # EnKF parameters
        self.alpha_0 = 1.
        self.rho = 0.1                   # \rho \in (0, 1)
        self.tau = 1 / self.rho + 1e-04  # \tau > 1/\rho
        self.eta = 1e-05                 # noise limit
        self.gamma = torch.eye(self.dim, dtype=torch_dtype)
        self.sqrt_gamma = torch.tensor(la.sqrtm(self.gamma)).tolist()
        self.P = Ensemble()  # stores momenta at t=0
        self.Q = Ensemble()  # stores shapes at t=1
        self.P_init = None  # for logging

        # termination criteria for the error
        self.atol = 1e-05
        self.max_iter = 50

        # internals for logging
        self._errors = []
        self._consensus = []

        self.dump_parameters()

    def predict(self):
        self.Q.clear()
        for p0 in self.P.ensemble:
            self.Q.append(self.shoot(p0))

    def correct(self):
        p_new = Ensemble()
        for p, q in zip(self.P.ensemble, self.Q.ensemble):
            p_new.append(p + self.gain(q))
        self.P = p_new

    def gain(self, w):
        cq = self.compute_cq()
        cp = self.compute_cp()

        p_update = torch.zeros(self.num_landmarks, self.dim)
        for k in range(self.num_landmarks):
            q_update = torch.matmul(cq[k, :, :], (self.target - w)[k])
            p_update[k] = torch.matmul(cp[k, :, :], q_update)

        return p_update

    def compute_cp(self):
        q_mean = self.Q.mean()
        p_mean = self.P.mean()

        cp = torch.zeros(self.num_landmarks, self.dim, self.dim)
        for j in range(self.ensemble_size):
            cp += torch.einsum('ij,ik->ijk', self.P.ensemble[j] - p_mean, self.Q.ensemble[j] - q_mean)

        return cp / (self.ensemble_size - 1)

    def compute_cq(self):
        """" Returns a regularised version of CQ. """
        lhs = self.rho * self.error_norm(self.target - self.Q.mean())
        cq = self.compute_cq_operator()

        k = 0
        max_iter_regularisation = 15
        alpha = self.alpha_0
        while k < max_iter_regularisation:
            # compute the operator of which we need the inverse
            cq_alpha_gamma_inv = torch.inverse(cq + alpha * self.gamma).tolist()

            # compute the error norm (rhs)
            q_cq_inv = torch.einsum('ijk,ij->ik', cq_alpha_gamma_inv, self.target - self.Q.mean())
            if alpha * self.error_norm(q_cq_inv) >= lhs:
                return cq_alpha_gamma_inv
            else:
                alpha *= 2
                k += 1
            return cq_alpha_gamma_inv
        raise EnKFError("!!! alpha failed to converge in {} iterations".format(max_iter_regularisation))

    def compute_cq_operator(self):
        q_ens = self.Q.ensemble
        q_mean = self.Q.mean()

        cq = torch.zeros((self.num_landmarks, self.dim, self.dim))
        for j in range(self.ensemble_size):
            cq += torch.einsum('ij,ik->ijk', q_ens[j] - q_mean, q_ens[j] - q_mean)
        cq /= self.ensemble_size - 1
        return cq

    def error_norm(self, x):
        # we use an \ell^2 norm of `\sqrt(Gamma)(mismatch)`
        prod_gamma_x = torch.einsum('ij,kj->ki', self.sqrt_gamma, x)
        return torch.sqrt(torch.einsum('ij,ij->', prod_gamma_x, prod_gamma_x))

    def run(self, p, target):
        k = 0
        self.P_init = p  # for logging
        self.P = p
        error = float("-inf")  # initial error
        while k < self.max_iter:
            print("Iteration ", k)
            self.predict()
            self.dump_mean(k, target)
            new_error = self.error_norm(self.target - self.Q.mean())
            self._errors.append(new_error)
            self._consensus.append(self.P.consensus())
            print("\t --> error norm: {}".format(new_error))
            if math.fabs(new_error-error) < self.atol:
                print("No improvement in residual, terminating filter")
                break
            elif new_error <= self.tau*self.eta:
                break
            else:
                self.correct()
                error = new_error
            k += 1
        self.dump_error()
        self.dump_consensus()
        return self.P, self.Q.mean().detach().numpy()

    def dump_error(self):
        po = open(self.log_dir + 'errors.pickle', 'w')
        pickle.dump(self._errors, po)
        po.close()

    def dump_consensus(self):
        po = open(self.log_dir + 'consensus.pickle', 'w')
        pickle.dump(self._consensus, po)
        po.close()

    def dump_mean(self, k, target):
        q_mean = self.Q.mean().detach().numpy()
        plot_landmarks(qs=q_mean, template=self.template, target=target,
                       file_name=self.log_dir + "PREDICTED_TARGET_iter={}".format(k))

    def dump_parameters(self):
        fh = open(self.log_dir + 'enkf_parameters.log', 'w')
        fh.write("max_iter: {}\n".format(self.max_iter))
        fh.write("num_landmarks: {}\n".format(self.num_landmarks))
        fh.write("Gamma: {}\n".format(self.gamma))
        fh.write("time-steps: {}\n".format(self.timesteps))
        fh.write("alpha_0: {}\n".format(self.alpha_0))
        fh.write("rho: {}\n".format(self.rho))
        fh.write("tau: {}\n".format(self.tau))
        fh.write("eta: {}\n".format(self.eta))
        fh.write("atol: {}\n".format(self.atol))
        fh.write("template: {}\n".format(self.template))
        fh.write("target: {}\n".format(self.target))
        if self.P_init:
            fh.write("P_init: {}\n".format(self.P_init))
        fh.close()
