from enkf import *
from lddmm import lddmm_forward, gauss_kernel
import utils
import pickle
import math

torch_dtype = torch.float32

ensemble_size = 8
num_landmarks = 20
timesteps = 20
dim = 2
pi = torch.tensor(math.pi)


sigma = torch.tensor([0.01], dtype=torch_dtype)
K = gauss_kernel(sigma=sigma)

# 1) define q0
q0_np, _, _ = utils.translation(num_landmarks)
q0 = torch.tensor(q0_np, dtype=torch_dtype, requires_grad=True)
q1 = torch.zeros(num_landmarks, dim, dtype=torch_dtype)

# 2) define q1 by shooting with a random initial momenta, and save this momenta in an Ensemble
#    (but scaled by a constant to produce a different shape!)

pe = Ensemble()
for j in range(ensemble_size):
    p0 = q0
    scale = .1
    for i in range(int(num_landmarks/2)):
        p0[i, :] *= scale*(1 + num_landmarks - i)/num_landmarks
        p0[num_landmarks-i-1, :] *= scale*(1 + num_landmarks - i)/num_landmarks
    q1 += lddmm_forward(p0, q0, K, timesteps)[-1][1]
    #p0 *= math.log(2**j)
    pe.append(p0)

q1 /= ensemble_size
plot_q(q1, "q1_truth")

# 3) set up and run Kalman filter
ke = EnsembleKalmanFilter(ensemble_size, q0, q1)
p_result = ke.run(pe)
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
