import torch
import math
import scipy.linalg as la

import utils
from ensemble import MomentumEnsemble, Ensemble

torch_dtype = torch.float32


class EnsembleKalmanFilter:

    def __init__(self, template, target, log_dir='./'):
        self.target = target
        self.template = template
        self.log_dir = log_dir
        self.dim = template.shape[1]
        self.num_landmarks = template.shape[0]

        # EnKF parameters
        self.alpha_0 = 2
        self.max_iter_regularisation = 1
        self.rho = 0.01                   # \rho \in (0, 1)
        self.tau = 1 / self.rho + 1e-04  # \tau > 1/\rho
        self.eta = 1e-05                 # noise limit
        self.gamma = torch.eye(self.dim, dtype=torch_dtype)
        self.root_gamma = torch.tensor(la.sqrtm(self.gamma))
        self.P = MomentumEnsemble()  # stores momenta at t=0
        self.Q = Ensemble()  # stores shapes at t=1

        # termination criteria for the error
        self.atol = 1e-05
        self.max_iter = 50

        # internals for logging
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
        print("\t!!! alpha regularisation failed to converge in {} iterations".format(self.max_iter_regularisation))
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

    def run(self, p, target):
        self.P = p

        # dump the target corresponding to the initial guess `p`
        target_initial_guess = self.P.forward(self.template)
        utils.plot_landmarks(qs=target_initial_guess.mean().detach().numpy(),
                             template=self.template,
                             target=target,
                             file_name=self.log_dir + f"PREDICTED_TARGET_INITIAL")

        k = 0
        error = float("-inf")  # initial error
        while k < self.max_iter:
            print("Iteration ", k)
            self.predict()
            self.dump_mean(k)
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
        return self.P

    def dump_error(self):
        utils.pdump(self._errors, self.log_dir + "errors.pickle")

    def dump_consensus(self):
        utils.pdump(self._consensus, self.log_dir + "consensus.pickle")

    def dump_mean(self, k):
        utils.pdump(self.Q.mean().detach().numpy(), self.log_dir + f"PREDICTED_TARGET_iter={k}.pickle")

    def dump_parameters(self):
        import os
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        fh = open(self.log_dir + 'enkf_parameters.log', 'w')
        fh.write("max_iter: {}\n".format(self.max_iter))
        fh.write("num_landmarks: {}\n".format(self.num_landmarks))
        fh.write("Gamma: {}\n".format(self.gamma))
        fh.write("alpha_0: {}\n".format(self.alpha_0))
        fh.write("rho: {}\n".format(self.rho))
        fh.write("tau: {}\n".format(self.tau))
        fh.write("eta: {}\n".format(self.eta))
        fh.write("atol: {}\n".format(self.atol))
        fh.write("template: {}\n".format(self.template))
        fh.write("target: {}\n".format(self.target))
        fh.close()
