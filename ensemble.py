import torch


class Ensemble:
    def __init__(self):
        self.ensemble = []

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
