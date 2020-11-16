import torch

import enkf_landmarks.utils as utils
from enkf_landmarks.lddmm import shoot

torch_dtype = torch.float32

time_steps = 10
num_landmarks = 20

template = torch.tensor(utils.circle(num_landmarks), dtype=torch_dtype, requires_grad=True)
target = torch.tensor(utils.circle(num_landmarks, shift=2., scale=2.), dtype=torch_dtype, requires_grad=True)

qs = shoot(template, target, time_steps)

utils.plot_landmarks('example_lddmm',
                     qs=qs,
                     template=template,
                     target=target,
                     title='Shooting via initial momentum.')


