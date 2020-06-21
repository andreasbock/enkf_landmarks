import torch
import utils
from lddmm import shoot

use_cuda = torch.cuda.is_available()
torchdtype = torch.float32
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'

nt = 10
nl = 20

Q0_np, Q1_np, test_name = utils.squeeze(nl)

Q0 = torch.tensor(Q0_np).float()
Q1 = torch.tensor(Q1_np).float()

q0 = Q0.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
q1 = Q1.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)

sigma = torch.tensor([0.01], dtype=torchdtype, device=torchdeviceId)
qs = shoot(q0, q1, nt, sigma)

utils.plot_q(qs)


