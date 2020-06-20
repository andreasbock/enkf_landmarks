# Standard imports

import os
import time
import numpy as np
#from tqdm import tqdm

# PyTorch and KeOps
from torch.autograd import grad
from pykeops.torch import Kernel, kernel_product
from pykeops.torch.kernel_product.formula import *
import pykeops.torch as pk

from torch.utils.tensorboard import SummaryWriter
import tensorboardX
import pylab as plt

# User-defined imports
import lib_landmarks

# torch type and device
use_cuda = torch.cuda.is_available()
torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
torchdtype = torch.float32

# PyKeOps counterpart
KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
KeOpsdtype = torchdtype.__str__().split('.')[1]  # 'float32'

####################################################################
# Define standard Gaussian kernel
# -------------------------------

def GaussKernel(sigma):
    def K(x, y, b):
        params = {
            'id': pk.Kernel('gaussian(x,y)'),
            'gamma': 1 / (sigma * sigma),
            'backend': 'auto'
        }
        return pk.kernel_product(params, x, y, b)
    return K

####################################################################
# ODE and Hamiltonian system
# --------------------------

def ForwardEulerIntegrator():
    def f(ODESystem, x0, nt):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = 1.0 / nt
        l = [x]
        for i in range(nt):
            xdot = ODESystem(*x)
            x = tuple(map(lambda x, xdot: x + xdot * dt, x, xdot))
            l.append(x)
        return l
    return f

def Hamiltonian(K):
    def H(p, q):
        return .5 * (p * K(q, q, p)).sum()
    return H

def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp
    return HS


#####################################################################
# Shooting approach
# -----------------

def Shooting(p0, q0, K, nt, Integrator=ForwardEulerIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt)

def LDDMMloss(K, nt, Q1):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K, nt)[-1]
        functional = ((q - Q1)*(q - Q1)).sum()
        return functional
    return loss

#####################################################################
# Define data attachment and LDDMM functional
# -------------------------------------------

def shoot(q0, q1, nt, sigma, p0=None, epochs=15):

    Kv = GaussKernel(sigma=sigma)
    loss = LDDMMloss(Kv, nt, q1)

    # initialize momentum vectors
    if p0 is None:
        p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId,
                         requires_grad=True)
    optimizer = torch.optim.LBFGS([p0], max_eval=100)

    print('Performing optimization...')
    start = time.time()

    def closure():
        optimizer.zero_grad()
        L = loss(p0, q0)
        print('loss', L.detach().cpu().numpy())
        L.backward()
        return L

    for i in range(epochs):
        optimizer.step(closure)
        Shooting(p0, q0, Kv, nt)

    end = time.time()
    print('Optimization took {:.2f}'.format(end - start))

    # final p/q solve
    pqs = Shooting(p0, q0, Kv, nt)
    qs = np.array([q.detach().numpy() for _, q in pqs])

    return qs
