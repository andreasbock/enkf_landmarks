import argparse
import torch

import src.ensemble as ensemble
import src.utils as utils

torch_dtype = torch.float32

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--log_dir', default=f"../TARGET_{utils.date_string()}/", help='Directory in which to dump data.')
parser.add_argument('--num_landmarks', type=int, default=50, help='Number of landmarks.')
parser.add_argument('--ensemble_size', type=int, default=10, help='Number of ensemble elements.')
parser.add_argument('--mean', type=float, default=0., help='Momentum mean.')
parser.add_argument('--std', type=float, default=.5, help='Momentum standard deviation.')
args = parser.parse_args()


def manufacture_from_normal_distribution(log_dir, ensemble_size, num_landmarks, mean, std):
    # 1) define template
    template_numpy = utils.circle(num_landmarks)
    template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)

    # 2) define target by shooting with a random initial momenta
    pe = ensemble.ensemble_normal(num_landmarks, ensemble_size, mean=mean, std=std)
    target = ensemble.target_from_momentum_ensemble(pe, template)

    # 3) Save the template, target and the initial ensemble & plot them
    pe.save(log_dir + "pe.pickle")
    utils.pdump(template, log_dir + "template.pickle")
    utils.pdump(target, log_dir + "target.pickle")
    utils.plot_landmarks(file_name=log_dir + "template_and_target", template=template, target=target)

    return template, target, pe


if __name__ == "__main__":
    manufacture_from_normal_distribution(log_dir=args.log_dir,
                                         ensemble_size=args.ensemble_size,
                                         num_landmarks=args.num_landmarks,
                                         mean=args.mean,
                                         std=args.std)
