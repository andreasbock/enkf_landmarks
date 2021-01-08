import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import src.utils as utils
from src.enkf import _initialise_distributed_pytorch
from src.lddmm import lddmm_forward, gauss_kernel

torch_dtype = torch.float32


def make_normal_momentum(num_landmarks, mean, std):
    _p = torch.distributions.normal.Normal(loc=mean, scale=std).sample([num_landmarks, 2])
    return _p.clone().detach().requires_grad_(True).type(dtype=torch_dtype)


def make_uniform_momentum(num_landmarks, low, high):
    _p = torch.distributions.uniform.Uniform(low, high).sample([num_landmarks, 2])
    return _p.clone().detach().requires_grad_(True).type(dtype=torch_dtype)


def manufacture_from_normal_distribution(ensemble_size,
                                         num_landmarks,
                                         mean,
                                         std,
                                         time_steps,
                                         landmark_size,
                                         log_dir=None):

    template = torch.tensor(utils.circle(num_landmarks), dtype=torch_dtype, requires_grad=True)
    p_ensemble_result = [torch.zeros(size=(num_landmarks, 2)) for _ in range(ensemble_size)]

    def _spawn(rank, ranks, p):
        _initialise_distributed_pytorch(rank, ranks)
        dist.all_gather(p_ensemble_result, p)

        # apply forward & reduce
        sigma = torch.tensor([landmark_size], dtype=torch_dtype)
        K = gauss_kernel(sigma=sigma)
        target = lddmm_forward(p, template, K, time_steps)[-1][1]
        dist.all_reduce(target)
        target /= ranks

        if log_dir and rank == 0:
            utils.pdump(p_ensemble_result, log_dir + "p_ensemble_truth.pickle")
            utils.pdump(template, log_dir + "template.pickle")
            utils.pdump(target, log_dir + "target.pickle")
            utils.plot_landmarks(file_name=log_dir + "template_and_target", template=template, target=target)

    processes = []
    for rank in range(ensemble_size):
        momentum = make_normal_momentum(num_landmarks, mean, std)
        p = Process(target=_spawn, args=(rank, ensemble_size, momentum))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return p_ensemble_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manufacture some random landmark configurations.')
    parser.add_argument('--log_dir', default=f"TARGET_{utils.date_string()}/", help='Directory in which to dump data.')
    parser.add_argument('--num_landmarks', type=int, default=150, help='Number of landmarks.')
    parser.add_argument('--ensemble_size', type=int, default=10, help='Number of ensemble elements.')
    parser.add_argument('--mean', type=float, default=0., help='Momentum mean.')
    parser.add_argument('--std', type=float, default=1., help='Momentum standard deviation.')
    parser.add_argument('--timesteps', type=int, default=1, help='Time steps for the forward map.')
    parser.add_argument('--landmark_size', type=float, default=1, help='Landmark size.')
    args = parser.parse_args()
    logger = utils.basic_logger(args.log_dir + '/manufactured_ensemble.log')

    for key, value in args.__dict__.items():
        logger.info(f"{key}: {value}")
        utils.pdump(value, args.log_dir + f"/{key}.pickle")

    manufacture_from_normal_distribution(ensemble_size=args.ensemble_size,
                                         num_landmarks=args.num_landmarks,
                                         mean=args.mean,
                                         std=args.std,
                                         time_steps=args.timesteps,
                                         landmark_size=args.landmark_size,
                                         log_dir=args.log_dir)
