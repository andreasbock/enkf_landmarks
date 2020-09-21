import torch

import ensemble
import utils

torch_dtype = torch.float32

ensemble_size = 10
num_landmarks = 10

log_dir = "example_target/"

# 1) define template
template_numpy = utils.circle(num_landmarks)
template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)

# 2) define target by shooting with a random initial momenta
p_target = ensemble.ensemble_normal(num_landmarks, ensemble_size)
target = ensemble.target_from_momentum_ensemble(p_target, template)

# 3) Save the template and target, a plot of them and the initial momentum
#    associated with the mapping from template -> target
p_target.save(log_dir + "p_target")
utils.pdump(template, log_dir + "template")
utils.pdump(target, log_dir + "target")
utils.plot_landmarks(file_name=log_dir + "template_and_target", template=template, target=target)

