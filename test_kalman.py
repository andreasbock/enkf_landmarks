from enkf import *
from lddmm import LDDMMForward, GaussKernel
import utils
import pickle

torch_dtype = torch.float32

ensemble_size = 8
num_landmarks = 20
timesteps = 20
dim = 2
pi = torch.tensor(math.pi)


sigma = torch.tensor([0.01], dtype=torch_dtype)
K = GaussKernel(sigma=sigma)

# 1) define q0 and q1, the latter by shooting from the distribution
Q0_np, Q1_np, test_name = utils.squeeze(num_landmarks)
q0 = torch.tensor(Q0_np, dtype=torch_dtype, requires_grad=True)
q1 = torch.zeros(num_landmarks, dim, dtype=torch_dtype)

for i in range(ensemble_size):
    p0 = torch.tensor([[torch.sin(i * 2 * pi / ensemble_size), torch.cos(i * 2 * pi / ensemble_size)]
                       for i in range(num_landmarks)],
                      dtype=torch_dtype, requires_grad=True)
    q1 += LDDMMForward(p0, q0, K, timesteps)[-1][1]
q1 /= ensemble_size

# 2) sample a random initial momentum (from the same distribution!)
pe = Ensemble()
we = Ensemble()
for i in range(ensemble_size):
    p0 = torch.tensor([[torch.cos(i * 2 * pi / ensemble_size), torch.sin(i * 2 * pi / ensemble_size)]
                       for i in range(num_landmarks)],
                      dtype=torch_dtype, requires_grad=True)
    q1 = LDDMMForward(p0, q0, K, timesteps)[-1][1]
    pe.append(p0)
    we.append(q1)

print("q0: ", q0, "\n")
print("q1: ", q1, "\n")

# 3) set up and run Kalman filter
ke = EnsembleKalmanFilter(ensemble_size, q0, q1)
p_result = ke.run(pe, we)
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
