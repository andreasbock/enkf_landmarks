import torch

import ensemble
import utils

torch_dtype = torch.float32

log_dir = f"TARGET_{utils.date_string()}/"
ensemble_size = 10
num_landmarks = 15

# 1) define template
template_numpy = utils.circle(num_landmarks)
template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)

# 2) define target by shooting with a random initial momenta
pe = ensemble.ensemble_normal(num_landmarks, ensemble_size, scale=.65)
target = ensemble.target_from_momentum_ensemble(pe, template)

# 3) Save the target, plot it and dump the initial momentum
pe.save(log_dir + "pe.pickle")
utils.pdump(target, log_dir + "target.pickle")
utils.plot_landmarks(file_name=log_dir + "template_and_target", template=template, target=target)

