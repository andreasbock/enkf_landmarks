import operator

import torch
import torch.distributed as dist

import src.utils as utils
from src.lddmm import lddmm_forward, gauss_kernel


torch_dtype = torch.float32


class Ensemble:
    def __init__(self, ensemble, rank):
        self.ensemble = ensemble
        self.rank = rank

    def mean(self):
        _mean = self.ensemble.clone()
        dist.all_reduce(_mean)
        _mean /= self.size()

    def consensus(self):
        normalised_ensemble = [e - self.mean() for e in self.ensemble]
        return torch.mean(torch.stack(normalised_ensemble), dim=0).norm()

    def save(self, file_name):
        utils.create_dir_from_path_if_not_exists(file_name)
        utils.pdump(self.ensemble, file_name)

    @staticmethod
    def load(file_name, rank, time_steps=10):
        e = MomentumEnsemble(time_steps=time_steps)
        e.ensemble = utils.pload(file_name)[rank]
        return e

    def perturb(self, alpha, op=operator.mul):
        for i in range(self.size()):
            self.ensemble[i] = op(self.ensemble, alpha)


class MomentumEnsemble(Ensemble):

    def __init__(self, time_steps=10):
        super().__init__()
        self.time_steps = time_steps

        # Shooting/ODE parameters
        sigma = torch.tensor([1], dtype=torch_dtype)
        self.k = gauss_kernel(sigma=sigma)

    def forward(self, template):
        q = lddmm_forward(self.ensemble, template, self.k, self.time_steps)[-1][1]
        return Ensemble(q)


def ensemble_normal(num_landmarks, ensemble_size, mean=0, std=1):
    pe = MomentumEnsemble()
    for j in range(ensemble_size):
        p0 = utils.sample_normal(num_landmarks, mean, std)
        pe.append(torch.tensor(p0, dtype=torch_dtype, requires_grad=True))
    return pe


def target_from_momentum_ensemble(pe, template, time_steps=10, landmark_size=1):
    # TODO: make this parallel
    sigma = torch.tensor([landmark_size], dtype=torch_dtype)
    K = gauss_kernel(sigma=sigma)
    target = torch.zeros(pe.element_dimension())
    for p in pe.ensemble:
        target += lddmm_forward(p, template, K, time_steps)[-1][1]
    return target / pe.size()
