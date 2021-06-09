from pathlib import Path

import utils
from lddmm import shoot

if __name__ == "__main__":

    # compute truths
    for truth_path in Path('data/').glob('LANDMARKS=*/TARGET*'):
        template = utils.pload(str(truth_path / 'template.pickle'))
        target = utils.pload(str(truth_path / 'target.pickle'))
        sigma = utils.pload(str(truth_path / 'std.pickle'))
        timesteps = utils.pload(str(truth_path / 'timesteps.pickle'))
        p_truth, q_truth = shoot(template, target, timesteps, sigma=sigma)
        utils.pdump(q_truth, str(truth_path / 'p_truth.pickle'))
        utils.plot_landmarks(file_name=str(truth_path / 'p_truth'),
                             qs=q_truth,
                             template=template,
                             target=target)