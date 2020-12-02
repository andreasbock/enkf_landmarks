import argparse
import torch

import src.utils as utils
from src.lddmm import lddmm_forward, gauss_kernel

torch_dtype = torch.float32

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--log_dir', default=f"../TARGET_{utils.date_string()}/", help='Directory in which to dump data.')
parser.add_argument('--num_landmarks', type=int, default=50, help='Number of landmarks.')
parser.add_argument('--ensemble_size', type=int, default=10, help='Number of ensemble elements.')
parser.add_argument('--mean', type=float, default=0., help='Momentum mean.')
parser.add_argument('--std', type=float, default=.5, help='Momentum standard deviation.')
args = parser.parse_args()


def ensemble_normal(num_landmarks, ensemble_size, mean=0, std=1):
    pe = []
    for j in range(ensemble_size):
        p0 = utils.sample_normal(num_landmarks, mean, std)
        pe.append(torch.tensor(p0, dtype=torch_dtype, requires_grad=True))
    return pe


def target_from_momentum_ensemble(pe, template, time_steps=10, landmark_size=1):
    # TODO: make this parallel
    sigma = torch.tensor([landmark_size], dtype=torch_dtype)
    K = gauss_kernel(sigma=sigma)
    target = torch.zeros(pe[0].size)
    for p in pe:
        target += lddmm_forward(p, template, K, time_steps)[-1][1]
    return target / len(pe)


def manufacture_from_normal_distribution(log_dir, ensemble_size, num_landmarks, mean, std):
    # 1) define template
    template_numpy = utils.circle(num_landmarks)
    template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)

    # 2) define target by shooting with a random initial momenta
    pe = ensemble_normal(num_landmarks, ensemble_size, mean=mean, std=std)
    target = target_from_momentum_ensemble(pe, template)

    # 3) Save the template, target and the initial ensemble & plot them
    utils.pdump(pe, log_dir + "pe.pickle")
    utils.pdump(template, log_dir + "template.pickle")
    utils.pdump(target, log_dir + "target.pickle")
    utils.plot_landmarks(file_name=log_dir + "template_and_target", template=template, target=target)

    return template, target, pe


if __name__ == "__main__":
    manufacture_from_normal_distribution(log_dir=args.log_dir,
                                         ensemble_size=args.ensemble_size,
                                         num_landmarks=args.num_landmarks,
                                         mean=args.mean,
                                         std=args.std)
