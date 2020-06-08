"""
=====================
Landmark registration
=====================

Example of ...

"""

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
# Registration parameters
# -----------------------

nl = 20  # number of landmarks
nt = 10  # number of time steps
deltat = 1.0  #total time interval
c_param = 1  # weight on || q(1) - q_1 ||_2
sig = 0.01 #sigma for landmarks
epochs = 5 #epochs for pytorch

#pringle
#squeeze
#triangle_flip
#pent_to_tri
#pent_to_square
#triangle_rot
#_test = lib_landmarks.criss_cross
_test = lib_landmarks.squeeze
Q0_np, Q1_np, test_name = _test(nl)

Q0 = torch.tensor(Q0_np).float()
Q1 = torch.tensor(Q1_np).float()

q0 = Q0.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
q1 = Q1.clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
sigma = torch.tensor([sig], dtype=torchdtype, device=torchdeviceId)


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
    def f(ODESystem, x0, nt, deltat=0.1):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
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
        #print("Gq", Gq)#Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp
    return HS


#####################################################################
# Shooting approach
# -----------------

def Shooting(p0, q0, K, nt, deltat, Integrator=ForwardEulerIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), nt, deltat)

def loss_function():

    def _loss(Q):
        return ((Q - Q1)*(Q - Q1)).sum()

    return _loss

def LDDMMloss(K, dataloss, penalty_weight):
    def loss(p0, q0):
        p, q = Shooting(p0, q0, K, nt, deltat)[-1]
        functional = penalty_weight * dataloss(q)

        # does not make sense to use initial conditions here, right?
        # should integrate? We don't need this term anyway, as we solve
        # Hamilton's equations already
        #functional += Hamiltonian(K)(p0, q0) # <- uncomment if you want!

        return functional
    return loss

#####################################################################
# Define data attachment and LDDMM functional
# -------------------------------------------

dataloss = loss_function()  # (... stuff like a split Kernel ...)
Kv = GaussKernel(sigma=sigma)
loss = LDDMMloss(Kv, dataloss, c_param)

######################################################################
# Perform optimization
# --------------------

# initialize momentum vectors
p0 = torch.zeros(q0.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

optimizer = torch.optim.LBFGS([p0], max_eval=100)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

print('performing optimization...')
start = time.time()

def closure():
    optimizer.zero_grad()
    L = loss(p0, q0)
    print('loss', L.detach().cpu().numpy())
    L.backward()
    return L


for i in range(epochs):
    optimizer.step(closure)
    Shooting(p0, q0, Kv, nt=nt, deltat=deltat)

end = time.time()
print('optimization took {:.2f}'.format(end - start))

# final p/q solve
pqs = Shooting(p0, q0, Kv, nt=nt, deltat=deltat)

qs = np.array([q.detach().numpy() for _, q in pqs])
lib_landmarks.plot_q(qs)

