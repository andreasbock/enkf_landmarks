import time
import torch

import utils
from lddmm import shoot

torch_dtype = torch.float32

time_steps = 10
num_landmarks = 20

template = torch.tensor(utils.circle(num_landmarks), dtype=torch_dtype, requires_grad=True)
target = torch.tensor(utils.circle(num_landmarks, shift=2., scale=2.), dtype=torch_dtype, requires_grad=True)

print("Shooting...")
start = time.time()
qs = shoot(template, target, time_steps)
end = time.time()
file_name = 'example_lddmm'

print(f"Done! Time elapsed: {end - start} seconds.")
print(f"Dumping output to {file_name}.pdf")

utils.plot_landmarks(file_name=file_name,
                     qs=qs,
                     template=template,
                     target=target,
                     title='Shooting via initial momentum.')


