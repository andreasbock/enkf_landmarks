import os
import sys
import time
import math
import torch
import logging
import scipy.linalg as la

import src.utils as utils
from src.ensemble import MomentumEnsemble, Ensemble

torch_dtype = torch.float32


class EnsembleKalmanFilter:

    def __init__(self, template, target, log_dir='./', max_iter=50):
        self.target = target
        self.template = template
        self.log_dir = log_dir
        self.dim = template.shape[1]
        self.num_landmarks = template.shape[0]

        # EnKF parameters mimicking those in:
        # Iglesias, Marco A. "A regularizing iterative ensemble Kalman method for PDE-constrained inverse problems."
        # Inverse Problems 32.2 (2016): 025002.
        # The are implemented here for future use, but in practice do not affect the current results.
        self.alpha_0 = 1
        self.max_iter_regularisation = 1
        self.rho = 0.01                  # \rho \in (0, 1)
        self.tau = 1 / self.rho + 1e-04  # \tau > 1/\rho
        self.eta = 1e-05                 # noise limit
        self.gamma = torch.eye(self.dim, dtype=torch_dtype)
        self.root_gamma = torch.tensor(la.sqrtm(self.gamma))
        self.P = MomentumEnsemble()  # stores momenta at t=0
        self.Q = Ensemble()  # stores shapes at t=1

        # termination criteria for the error
        self.atol = 1e-05
        self.max_iter = max_iter

        # internals for logging
        self.logger = self.setup_logger()
        self._errors = []
        self._consensus = []
        self.dump_parameters()

    def predict(self):
        self.Q = self.P.forward(self.template)

    def correct(self):
        q_mean = self.Q.mean()
        cq = self._compute_cq(q_mean)
        cp = self._compute_cp(q_mean)

        p_new = MomentumEnsemble()
        for p, q in zip(self.P.ensemble, self.Q.ensemble):
            p_new.append(p + self.gain(q, cp, cq))
        self.P = p_new

    def gain(self, w, cp, cq):
        p_update = torch.zeros(self.num_landmarks, self.dim)
        for k in range(self.num_landmarks):
            q_update = torch.matmul(cq[k, :, :], (self.target - w)[k])
            p_update[k] = torch.matmul(cp[k, :, :], q_update)
        return p_update

    def _compute_cp(self, q_mean):
        p_mean = self.P.mean()
        cp = torch.zeros(self.num_landmarks, self.dim, self.dim)
        for p, q in zip(self.P.ensemble, self.Q.ensemble):
            cp += torch.einsum('ij,ik->ijk', p - p_mean, q - q_mean)
        return cp / (self.Q.size() - 1)

    def _compute_cq(self, q_mean):
        """" Returns a regularised version of CQ. """
        lhs = self.rho * self.error_norm(self.target - q_mean)
        cq = self._compute_cq_operator(q_mean)

        k = 0
        alpha = self.alpha_0
        while k < self.max_iter_regularisation:
            # compute the operator of which we need the inverse
            cq_alpha_gamma_inv = torch.inverse(cq + alpha * self.gamma)
            # compute the error norm (rhs)
            q_cq_inv = torch.einsum('ijk,ij->ik', cq_alpha_gamma_inv, self.target - q_mean)
            if alpha * self.error_norm(q_cq_inv) >= lhs:
                return cq_alpha_gamma_inv
            else:
                alpha *= 2
                k += 1
        return cq_alpha_gamma_inv

    def _compute_cq_operator(self, q_mean):
        cq = torch.zeros((self.num_landmarks, self.dim, self.dim))
        for q in self.Q.ensemble:
            cq += torch.einsum('ij,ik->ijk', q - q_mean, q - q_mean)
        return cq / (self.Q.size() - 1)

    def error_norm(self, x):
        # we use an \ell^2 norm of `\sqrt(Gamma)(mismatch)`
        prod_gamma_x = torch.einsum('ij,kj->ki', self.root_gamma, x)
        return torch.sqrt(torch.einsum('ij,ij->', prod_gamma_x, prod_gamma_x))

    def run(self, p, target, regularisation=1):
        self.P = p
        self.alpha_0 = regularisation

        # dump the target corresponding to the initial guess `p`
        target_initial_guess = self.P.forward(self.template)
        utils.plot_landmarks(qs=target_initial_guess.mean().detach().numpy(),
                             template=self.template,
                             target=target,
                             file_name=self.log_dir + "template_and_target")
        start = time.time()
        k = 0
        error = float("-inf")  # initial error
        while k < self.max_iter:
            self.predict()
            self.dump_mean(k)
            new_error = self.error_norm(self.target - self.Q.mean())
            self._errors.append(new_error)
            self._consensus.append(self.P.consensus())

            self.logger.info("Iteration {} | Error norm: {}".format(k, new_error))
            if math.isnan(new_error):
                self.logger.critical("Error is NaN (regularisation issues?), terminating filter.")
                break
            elif math.fabs(new_error - error) < self.atol:
                self.logger.info("No improvement in residual, terminating filter")
                break
            elif new_error <= self.tau*self.eta:
                break
            else:
                self.correct()
                error = new_error
                k += 1
        end = time.time()
        time_elapsed = time.strftime('%H:%M:%S', time.gmtime(end - start))
        utils.pdump(time_elapsed, self.log_dir + "run_time.pickle")
        self.logger.info(f"Filter run time: {time_elapsed}. Logged to {self.log_dir + 'run_time.pickle'}.")

        self.dump_error()
        self.dump_consensus()

        return self.P

    def dump_error(self):
        utils.pdump(self._errors, self.log_dir + "errors.pickle")

    def dump_consensus(self):
        utils.pdump(self._consensus, self.log_dir + "consensus.pickle")

    def dump_mean(self, k):
        utils.pdump(self.Q.mean().detach().numpy(), self.log_dir + f"PREDICTED_TARGET_iter={k}.pickle")

    def dump_parameters(self):
        self.logger.info("max_iter: {}".format(self.max_iter))
        self.logger.info("num_landmarks: {}".format(self.num_landmarks))
        self.logger.info("Gamma: {}".format(self.gamma))
        self.logger.info("alpha_0: {} (regularising parameter)".format(self.alpha_0))
        self.logger.info("rho: {}".format(self.rho))
        self.logger.info("tau: {}".format(self.tau))
        self.logger.info("eta: {}".format(self.eta))
        self.logger.info("atol: {}  (error tolerance)".format(self.atol))

    def setup_logger(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        logger_name = self.log_dir + '/enkf.log'
        logger = logging.getLogger(logger_name)

        logger.setLevel(logging.INFO)
        format_string = "%(asctime)s [%(levelname)s]: %(message)s"
        log_format = logging.Formatter(format_string, "%Y-%m-%d %H:%M:%S")

        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

        # Creating and adding the file handler
        file_handler = logging.FileHandler(logger_name)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        return logger
