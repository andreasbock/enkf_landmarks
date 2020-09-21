import torch
import pickle
import operator

import utils
from lddmm import lddmm_forward, gauss_kernel


torch_dtype = torch.float32


class Ensemble:
    def __init__(self):
        self.ensemble = []
        self.spatial_dimension = 2

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
        return torch.mean(torch.stack(normalised_ensemble), dim=0)

    def save(self, file_name):
        utils.create_dir_from_path_if_not_exists(file_name)
        po = open(f"{file_name}.pickle", "wb")
        pickle.dump(self.ensemble, po)
        po.close()

    def load(self, file_name):
        po = open(f"{file_name}.pickle", "wb")
        self.ensemble = pickle.load(po)

    def perturb(self, alpha, op=operator.add):
        for i in range(self.size()):
            self.ensemble[i] = op(self.ensemble[i], alpha[i])


def ensemble_normal(num_landmarks, ensemble_size):
    pe = Ensemble()
    for j in range(ensemble_size):
        p0_np = utils.sample_normal(num_landmarks, 0, 1)
        p0 = torch.tensor(p0_np, dtype=torch_dtype, requires_grad=True)
        pe.append(p0)
    return pe


def target_from_momentum_ensemble(pe, template, time_steps=10, landmark_size=1):
    sigma = torch.tensor([landmark_size], dtype=torch_dtype)
    K = gauss_kernel(sigma=sigma)
    target = torch.zeros((pe.size(), pe.spatial_dimension))
    for p in pe.ensemble:
        # Example 2: Normal distribution
        target += lddmm_forward(p, template, K, time_steps)[-1][1]
    return target / pe.size()
