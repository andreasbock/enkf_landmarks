import os
import time

import math
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import scipy.linalg as la

import src.utils as utils
from src.lddmm import lddmm_forward, gauss_kernel

torch_dtype = torch.float32


class EnsembleKalmanFilter:

    def __init__(self,
                 template,
                 target,
                 log_dir='./'):
        self.log_dir = log_dir
        self.logger = utils.basic_logger(self.log_dir + '/enkf.log')

        self.target = target
        self.template = template
        self.num_landmarks, self.dim = template.shape
        self.logger.info("num_landmarks: {}".format(self.num_landmarks))

        # EnKF parameters mimicking those in:
        # Iglesias, Marco A. "A regularizing iterative ensemble Kalman method for PDE-constrained inverse problems."
        # Inverse Problems 32.2 (2016): 025002.
        # The are implemented here for future use, but in practice do not affect the current results.
        self.max_iter_regularisation = 1
        self.rho = 0.01                  # \rho \in (0, 1)
        self.tau = 1 / self.rho + 1e-04  # \tau > 1/\rho
        self.eta = 1e-05                 # noise limit
        self.gamma = torch.eye(self.dim, dtype=torch_dtype)
        self.root_gamma = torch.tensor(la.sqrtm(self.gamma))
        self.atol = 1e-05

        # ODE parameters
        default_sigma = 1
        self.sigma = torch.tensor([default_sigma], dtype=torch_dtype)
        self.kernel = gauss_kernel(sigma=self.sigma)
        self.time_steps = None

        self.P = None  # stores momenta at t=0
        self.Q = None  # stores shapes at t=1

        # parallelisation params
        self.rank = None
        self.ranks = None

        self.logger.info("atol: {}  (error tolerance)".format(self.atol))
        self.logger.info("rho: {}".format(self.rho))
        self.logger.info("tau: {}".format(self.tau))
        self.logger.info("eta: {}".format(self.eta))
        self.logger.info("Gamma: {}".format(self.gamma))
        self.logger.info("kernel size: {}".format(self.sigma))

    def predict(self):
        self.Q = self._forward()
        return self._mean(self.Q)

    def correct(self, q_mean, p_mean, regularisation):
        # do these on master only & share
        cq = self._cov_qq(q_mean, regularisation)
        cp = self._cov_pq(q_mean, p_mean)
        with torch.no_grad():
            self.P += self.gain(cp, cq)

    def gain(self, cp, cq):
        p_update = torch.zeros(self.num_landmarks, self.dim)
        for k in range(self.num_landmarks):
            q_update = torch.matmul(cq[k, :, :], (self.target - self.Q)[k])
            p_update[k] = torch.matmul(cp[k, :, :], q_update)
        return p_update

    def _mean(self, tensor):
        _tensor = tensor.clone()
        dist.all_reduce(_tensor)
        return _tensor / self.ranks

    def _cov_pq(self, q_mean, p_mean):
        cp = torch.einsum('ij,ik->ijk', self.P - p_mean, self.Q - q_mean)
        dist.all_reduce(cp)
        return cp / (self.ranks - 1)

    def _cov_qq(self, q_mean, alpha):
        """" Returns a regularised version of CQ. """
        cqq_alpha_gamma_inv = torch.zeros(self.num_landmarks, self.dim, self.dim)
        cqq = self._cov_qq_operator(q_mean)

        lhs = self.rho * self.error_norm(self.target - q_mean)
        _alpha = torch.tensor(alpha)

        k = 0
        while k < self.max_iter_regularisation:
            # compute the operator of which we need the inverse
            cqq_alpha_gamma_inv = torch.inverse(cqq + _alpha * self.gamma)
            # compute the error norm (rhs)
            q_cqq_inv = torch.einsum('ijk,ij->ik', cqq_alpha_gamma_inv, self.target - q_mean)
            if _alpha * self.error_norm(q_cqq_inv) >= lhs:
                return cqq_alpha_gamma_inv
            else:
                _alpha *= 2
                k += 1
        return cqq_alpha_gamma_inv

    def _cov_qq_operator(self, q_mean):
        cq = torch.einsum('ij,ik->ijk', self.Q - q_mean, self.Q - q_mean)
        dist.all_reduce(cq)
        return cq / (self.ranks - 1)

    def error_norm(self, x):
        # we use an l^2 norm of `\sqrt(Gamma)(mismatch)`
        prod_gamma_x = torch.einsum('ij,kj->ki', self.root_gamma, x)
        return torch.sqrt(torch.einsum('ij,ij->', prod_gamma_x, prod_gamma_x))

    def run(self, p_ensemble, regularisation=1, time_steps=10, max_iter=50):
        self.ranks = len(p_ensemble)
        self.time_steps = time_steps

        self.logger.info("alpha: {} (regularising parameter)".format(regularisation))
        self.logger.info("time steps: {}".format(time_steps))
        self.logger.info("max_iter: {}".format(max_iter))

        start = time.time()
        processes = []
        p_ensemble_result = [torch.zeros(size=p_ensemble[0].size()) for _ in p_ensemble]
        for rank in range(self.ranks):
            process = Process(target=self._run, args=(p_ensemble, rank, p_ensemble_result, regularisation, max_iter))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        end = time.time()
        time_elapsed = time.strftime('%H:%M:%S', time.gmtime(end - start))
        utils.pdump(time_elapsed, self.log_dir + "/run_time.pickle")
        self.logger.info(f"Filter run time: {time_elapsed}. Logged to {self.log_dir + 'run_time.pickle'}.")

        return p_ensemble_result

    def _run(self, p_ensemble, rank, p_ensemble_result, regularisation, max_iter):
        _initialise_distributed_pytorch(rank, self.ranks)
        self.rank = rank
        self.P = p_ensemble[rank].clone().detach().requires_grad_(True)
        misfits = []

        k = 0
        error = float("-inf")  # initial error
        while k < max_iter:
            q_mean = self.predict()
            new_error = self.error_norm(self.target - q_mean)
            if self.is_master():
                misfits.append(new_error)
            self.logger_info("Iteration {} | Error norm: {}".format(k, new_error))

            if self.has_converged(new_error, error):
                break
            else:
                p_mean = self._mean(self.P)
                self.dump_means(k, q_mean, p_mean)
                self.correct(q_mean, p_mean, regularisation)
                error = new_error
                k += 1

        if self.is_master():
            utils.pdump(misfits, self.log_dir + "/misfits.pickle")

        dist.all_gather(p_ensemble_result, self.P)
        return p_ensemble_result

    def has_converged(self, new_error, error):
        if math.isnan(new_error):
            self.logger_info("Error is NaN (regularisation issue?), terminating filter.")
            return True
        elif math.fabs(new_error - error) < self.atol:
            self.logger_info("No improvement in residual, terminating filter.")
            return True
        elif new_error <= self.tau * self.eta:
            self.logger_info(f"Error {new_error} below tolerance {self.tau * self.eta}, terminating filter.")
            return True
        else:
            return False

    def _forward(self):
        return lddmm_forward(self.P, self.template, self.kernel, self.time_steps)[-1][1]

    def dump_means(self, k, q_mean, p_mean):
        if self.is_master():
            utils.pdump(q_mean.detach().numpy(), self.log_dir + f"/Q_mean_iter={k}.pickle")
            utils.pdump(p_mean.detach().numpy(), self.log_dir + f"/P_mean_iter={k}.pickle")

    def logger_info(self, msg):
        if self.is_master():
            self.logger.info(msg)

    def is_master(self):
        return self.rank == 0 or self.rank is None


def _initialise_distributed_pytorch(rank, ranks, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
    dist.init_process_group(backend=backend, rank=rank, world_size=ranks)
