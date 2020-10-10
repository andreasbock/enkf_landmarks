import numpy as np
from torch.autograd import grad
from pykeops.torch.kernel_product.formula import *
import pykeops.torch as pk

torch_dtype = torch.float32


####################################################################
# Define standard Gaussian kernel
# -------------------------------

def gauss_kernel(sigma):
    def k(x, y, b):
        params = {
            'id': pk.Kernel('gaussian(x,y)'),
            'gamma': 1 / (sigma * sigma),
            'backend': 'auto'
        }
        return pk.kernel_product(params, x, y, b)

    return k


####################################################################
# ODE and Hamiltonian system
# --------------------------

def forward_euler():
    def f(ode_system, x0, nt):
        x = tuple(map(lambda _x: _x.clone(), x0))
        dt = 1.0 / nt
        xs = [x]
        for i in range(nt):
            x_dot = ode_system(*x)
            x = tuple(map(lambda _x, _xd: _x + _xd * dt, x, x_dot))
            xs.append(x)
        return xs
    return f


def hamiltonian(k):
    def h(p, q):
        return .5 * (p * k(q, q, p)).sum()
    return h


def hamiltonian_system(k):
    h = hamiltonian(k)

    def hs(p, q):
        gp, gq = grad(h(p, q), (p, q), create_graph=True)
        return -gq, gp

    return hs


#####################################################################
# Shooting approach
# -----------------

def lddmm_forward(p0, q0, k, nt, integrator=forward_euler()):
    return integrator(hamiltonian_system(k), (p0, q0), nt)


#####################################################################
# Shooting
# -------------------------------------------

def lddmm_loss(k, nt, q1):
    def loss(p0, q0):
        p, q = lddmm_forward(p0, q0, k, nt)[-1]
        functional = ((q - q1) * (q - q1)).sum()
        return functional

    return loss


def shoot(q0, q1, nt, sigma, p0=None, epochs=15):
    k = gauss_kernel(sigma=sigma)
    loss = lddmm_loss(k, nt, q1)

    # initialize momentum vectors
    if p0 is None:
        p0 = torch.zeros(q0.shape, dtype=torch_dtype, requires_grad=True)
    optimizer = torch.optim.LBFGS([p0], max_eval=100)

    def closure():
        optimizer.zero_grad()
        _loss = loss(p0, q0)
        print('loss', _loss.detach().cpu().numpy())
        _loss.backward()
        return _loss

    for _ in range(epochs):
        optimizer.step(closure)
        lddmm_forward(p0, q0, k, nt)

    # final p/q solve
    pqs = lddmm_forward(p0, q0, k, nt)
    qs = np.array([q.detach().numpy() for _, q in pqs])
    return qs
