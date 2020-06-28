import torch


class Ensemble:
    def __init__(self):
        self.ensemble = []

    def mean(self):
        if len(self.ensemble) < 1:
            raise ValueError("Cannot take mean of empty ensemble")
        else:
            result = torch.zeros(size=self.ensemble[0].size())
            for e in self.ensemble:
                result += e
            return result / len(self.ensemble)
            return torch.mean(torch.stack(self.ensemble), dim=0)

    def append(self, el):
        self.ensemble.append(el)

    def clear(self):
        self.ensemble = []
