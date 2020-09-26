import torch
import operator

import utils
from lddmm import lddmm_forward, gauss_kernel


torch_dtype = torch.float32


class Ensemble:
    def __init__(self):
        self.ensemble = []

    def size(self):
        return len(self.ensemble)

    def mean(self):
        if len(self.ensemble) < 1:
            raise ValueError("Cannot take mean of empty ensemble")
        else:
            return torch.mean(torch.stack(self.ensemble), dim=0)

    def append(self, el):
        self.ensemble.append(el)

    def clear(self):
        self.ensemble = []

    def consensus(self):
        normalised_ensemble = [e - self.mean() for e in self.ensemble]
        return torch.mean(torch.stack(normalised_ensemble), dim=0).norm()

    def save(self, file_name):
        utils.create_dir_from_path_if_not_exists(file_name)
        utils.pdump(self.ensemble, file_name)

    @staticmethod
    def load(file_name):
        ens = Ensemble()
        ens.ensemble = utils.pload(file_name)
        return ens

    def perturb(self, alpha, op=operator.mul):
        for i in range(self.size()):
            self.ensemble[i] = op(self.ensemble[i], alpha[i])

    def element_dimension(self):
        if len(self.ensemble) == 0:
            raise Exception("Cannot get dimension of Ensemble with no elements!")
        return self.ensemble[0].size()


def ensemble_normal(num_landmarks, ensemble_size, scale=1):
    pe = Ensemble()
    for j in range(ensemble_size):
        p0_np = scale * utils.sample_normal(num_landmarks, 0, 1)
        p0 = torch.tensor(p0_np, dtype=torch_dtype, requires_grad=True)
        pe.append(p0)
    return pe


def target_from_momentum_ensemble(pe, template, time_steps=10, landmark_size=1):
    sigma = torch.tensor([landmark_size], dtype=torch_dtype)
    K = gauss_kernel(sigma=sigma)
    target = torch.zeros(pe.element_dimension())
    for p in pe.ensemble:
        # Example 2: Normal distribution
        target += lddmm_forward(p, template, K, time_steps)[-1][1]
    return target / pe.size()
