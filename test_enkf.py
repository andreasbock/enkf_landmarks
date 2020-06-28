from enkf import *
from lddmm import lddmm_forward, gauss_kernel
import utils
import pickle
import math
import numpy as np

torch_dtype = torch.float32
ensemble_size = 10
num_landmarks = 20
timesteps = 10
dim = 2

sigma = torch.tensor([1], dtype=torch_dtype)
K = gauss_kernel(sigma=sigma)

# 1) define q0
q0_np, _, _ = utils.translation(num_landmarks)
q0 = torch.tensor(q0_np, dtype=torch_dtype, requires_grad=True)
q1 = torch.zeros(num_landmarks, dim, dtype=torch_dtype)

# 2) define q1 by shooting with a random initial momenta, and save this momenta in an Ensemble
#    (but scaled by a constant to produce a different shape!)
pe = Ensemble()
for j in range(ensemble_size):
    # Example 1
    #p0x = utils.sample_vonMises(num_landmarks, kappa=0.001)
    #p0y = utils.sample_vonMises(num_landmarks, kappa=0.001)
    #p0_np = (np.stack((p0x, p0y), axis=1))
    #p0 = torch.tensor(p0_np, dtype=torch_dtype, requires_grad=True)
    #q1 += lddmm_forward(p0, q0, K, timesteps)[-1][1]

    # perturb the ensemble
    #p0x = utils.sample_vonMises(num_landmarks, kappa=0.00001)
    #p0y = utils.sample_vonMises(num_landmarks, kappa=0.00001)
    #p0_np = np.fabs(np.stack((p0x, p0y), axis=1))
    #p0 = torch.tensor(np.random.uniform(-5, 5) * p0_np, dtype=torch_dtype, requires_grad=True)
    #pe.append(p0)

    # Example 2
    #for i in range(int(num_landmarks/2)):
    #    scale = 5*math.exp(-(i-int(num_landmarks/2))**2/4)
    #    p0[i, :] *= scale*(1 + num_landmarks - i)/num_landmarks
    #    p0[num_landmarks-i-1, :] *= scale*(1 + num_landmarks - i)/num_landmarks
    p0_np = utils.sample_normal(num_landmarks, 0, 1)
    p0 = torch.tensor(p0_np, dtype=torch_dtype, requires_grad=True)
    q1 += lddmm_forward(p0, q0, K, timesteps)[-1][1]

    # perturb the ensemble
    p0_np = utils.sample_normal(num_landmarks, 0, 1)
    p0 = torch.tensor(np.random.uniform(-5, 5) * p0_np, dtype=torch_dtype, requires_grad=True)
    pe.append(p0)

q1 /= ensemble_size
print("q1: ", q1)
plot_q(q1, "q1_truth")

# 3) set up and run Kalman filter
ke = EnsembleKalmanFilter(ensemble_size, q0, q1)
p_result = ke.run(pe, q1=q1)
ke.dump_parameters()

# 4) use the momentum we found to compute the mean observation
W = Ensemble()
for e in p_result.ensemble:
    W.append(ke.shoot(e)[1])
w = W.mean()

# 5) dump to file
po = open("p_result.pickle", "wb")
pickle.dump(p_result, po)
po.close()

po = open("w_mean.pickle", "wb")
pickle.dump(w, po)
po.close()
