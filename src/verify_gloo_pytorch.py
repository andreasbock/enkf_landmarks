import os
import time
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

import src.utils as utils
from src.lddmm import shoot

torch_dtype = torch.float32

time_steps = 10
num_landmarks = 20

template = torch.tensor(utils.circle(num_landmarks), dtype=torch_dtype, requires_grad=True)
target = torch.tensor(utils.circle(num_landmarks, shift=2., scale=2.), dtype=torch_dtype, requires_grad=True)


def run(rank, size):
    print(f"Rank: {rank} of {size}\n")
    qs = shoot(template, target, time_steps)
    return qs


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    print("Remember to run 'export GLOO_SOCKET_IFNAME=en0'!")

    size = 4
    processes = []

    starttime = time.time()
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print('That took {} seconds'.format(time.time() - starttime))
