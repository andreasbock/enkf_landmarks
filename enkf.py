import torch
import math
import scipy.linalg as la

from lddmm import LDDMMForward, GaussKernel
from ensemble import Ensemble

torch_dtype = torch.float32


class EnsembleKalmanFilter:

    def __init__(self, ensemble_size, q0, q1, log_dir='./'):
        self.ensemble_size = ensemble_size
        self.q0 = q0
        self.q1 = q1
        self.log_dir = log_dir

        self.q_dim = q0.shape[1]
        self.p_dim = q0.shape[1]
        self.num_landmarks = q0.shape[0]

        # Shooting/ODE parameters
        sigma = torch.tensor([0.01], dtype=torch_dtype)
        k = GaussKernel(sigma=sigma)
        self.timesteps = 20
        self.shoot = lambda p0: LDDMMForward(p0, self.q0, k, self.timesteps)[-1]

        # EnKF parameters
        self.alpha_0 = 1.
        self.rho = 1                     # \rho \in (0, 1)
        self.tau = 1 / self.rho + 1e-04  # \tau > 1/\rho
        self.eta = 1e-03                 # noise limit
        self.gamma = torch.eye(self.q_dim, dtype=torch_dtype)
        self.sqrt_gamma = torch.tensor(la.sqrtm(self.gamma))
        self.W = None
        self.P = None
        # termination criteria for the error
        self.atol = 1e-05
        self.P_init = None
        self.max_iter = 10**5

    def predict(self):
        self.W.clear()
        for e in self.P.ensemble:
            self.W.append(self.shoot(e)[1])

    def correct(self):
        p_new = Ensemble()
        for i, (p, w) in enumerate(zip(self.P.ensemble, self.W.ensemble)):
            p_new.append(p + self.gain(w))

        self.P = p_new

    def gain(self, w):
        cw = self.compute_cw_op()
        cp = self.compute_cp()

        p_update = torch.zeros(self.num_landmarks, self.p_dim)
        for k in range(self.num_landmarks):
            q_update = torch.matmul(cw[k, :, :], (self.q1 - w)[k])
            p_update[k] = torch.matmul(cp[k, :, :], q_update)

        return p_update

    def compute_cw(self):
        w_ens = self.W.ensemble
        w_mean = self.W.mean()

        cw = torch.zeros(self.num_landmarks, self.q_dim, self.q_dim)
        for j in range(self.ensemble_size):
            cw += torch.einsum('ij,ik->ijk', w_ens[j] - w_mean, w_ens[j] - w_mean)
        cw /= (self.ensemble_size - 1)
        return cw

    def compute_cp(self):
        w_mean = self.W.mean()
        p_mean = self.P.mean()

        cp = torch.zeros(self.num_landmarks, self.p_dim, self.q_dim)
        for j in range(self.ensemble_size):
            cp += torch.einsum('ij,ik->ijk', self.P.ensemble[j] - p_mean, self.W.ensemble[j] - w_mean)

        return cp / (self.ensemble_size - 1)

    def compute_cw_op(self):
        lhs = self.rho * self.error_norm(self.q1 - self.W.mean())
        cw = self.compute_cw()

        k = 0
        max_iter = 10**3
        alpha = self.alpha_0
        while k < max_iter:

            # compute the operator of which we need the inverse
            cw_alpha_gamma_inv = torch.inverse(cw + alpha * self.gamma)

            # compute the error norm (rhs)
            w_cw_inv = torch.einsum('ijk,ij->ik', cw_alpha_gamma_inv, self.q1 - self.W.mean())
            rhs = self.error_norm(w_cw_inv)
            if alpha * rhs >= lhs:
                return cw_alpha_gamma_inv
            else:
                alpha *= 2
                k += 1

        raise ValueError("!!! alpha failed to converge in {} iterations".format(max_iter))

    def error_norm(self, x):
        # assumes we use an \ell^2 norm of `\sqrt(Gamma)(mismatch)`
        inner_norm = torch.einsum("ij,kj->ki", self.sqrt_gamma, x)
        err_norm = torch.sqrt(torch.einsum('ij,ij->', inner_norm, inner_norm))
        return err_norm

    def run(self, p, w):
        k = 0
        self.P_init = p  # for logging
        self.P = p
        self.W = w
        err = float("-inf")
        while k < self.max_iter:
            print("Iter ", k)
            self.predict()
            self.dump_mean()
            n_err = self.error_norm(self.q1 - self.W.mean())
            print("\t\t --> error norm: {}".format(n_err))
            if math.fabs(n_err-err) < self.atol:
                print("No improvement in residual, terminating filter")
                break
            elif n_err <= self.tau*self.eta:
                break
            else:
                self.correct()
                err = n_err
            k += 1
        return self.P

    def dump_mean(self):
        print("W_mean: {}".format(self.W.mean()))

    def dump_parameters(self):
        import os
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        fh = open(self.log_dir + 'parameters.log', 'w')
        fh.write("max_iter: {}\n".format(self.max_iter))
        fh.write("num_landmarks: {}\n".format(self.num_landmarks))
        fh.write("Gamma: {}\n".format(self.gamma))
        fh.write("timesteps: {}\n".format(self.timesteps))
        fh.write("alpha_0: {}\n".format(self.alpha_0))
        fh.write("rho: {}\n".format(self.rho))
        fh.write("tau: {}\n".format(self.tau))
        fh.write("eta: {}\n".format(self.eta))
        fh.write("atol: {}\n".format(self.atol))
        fh.write("q0: {}\n".format(self.q0))
        fh.write("q1: {}\n".format(self.q1))
        if self.P_init:
            fh.write("P_init: {}\n".format(self.P_init))
        fh.close()
