import torch


class Ensemble:
    def __init__(self):
        self.ensemble = []

    def mean(self):
        if len(self.ensemble) < 1:
            raise ValueError("Cannot take mean of empty ensemble")
        else:
            return torch.mean(torch.stack(self.ensemble))

    def append(self, ne):
        self.ensemble.append(ne)

    def clear(self):
        self.ensemble = []
