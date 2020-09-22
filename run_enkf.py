import numpy as np

from enkf import *
import utils
import sys

data_dir = "./TARGET_1/"
# where to fetch template
if data_dir is None and len(sys.argv) < 1:
    raise Exception("Must provide a data directory from which to load momentum and target!")

# where to dump results
log_dir = f"EXAMPLE_{utils.date_string()}/"

# 1) load initial ensemble and target from file
pe = Ensemble.load(data_dir + "pe.pickle")
target = utils.pload(data_dir + "target.pickle")

# 2) make a template to start from
template_numpy = utils.circle(len(target))
template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)
utils.plot_landmarks(file_name=log_dir + "template_and_target",
                     template=template,
                     target=target)

# 3) perturb ensemble by either
#   A: adding noise:      p = p_target + \eta, \eta noise
#   B: multiplying noise: p = p_target + \eta, \eta noise
#       Note: this can be at ensemble level or element level!
low = -10
high = 10
alpha = [np.random.uniform(low, high) for _ in pe.ensemble]
pe.perturb(alpha)

# 4) set up and run Kalman filter
ke = EnsembleKalmanFilter(template, target, log_dir=log_dir)
p, q = ke.run(pe, target)
utils.pdump(p, log_dir + "p_result.pickle")
utils.pdump(alpha, log_dir + "weight_vector.pickle")
utils.pdump(q, log_dir + "final_q_mean.pickle")

# TODO: check that `q` above is the same as `q` below!
Q = Ensemble()
for e in p.ensemble:
    Q.append(ke.shoot(e)[1])
w = Q.mean()
