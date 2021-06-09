import numpy as np
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def compute_relative_error(truth):
    def _compute_relative_error(approx):
        return np.linalg.norm(approx - truth) / np.linalg.norm(truth)
    return _compute_relative_error


if __name__ == "__main__":
    # plot the misfits for each target/landmark/ensemble size together
    for path in Path('RESULTS_ENKF/').glob('LANDMARKS=*/TARGET*/ENSEMBLE_SIZE=*'):
        path_string = str(path)

        # fix this:
        landmarks = re.search('LANDMARKS=([0-9]+)', path_string).group()
        ensemble_size = re.search('ENSEMBLE_SIZE=([0-9]+)', path_string).group()
        target = re.search('TARGET_(.)*/', path_string).group()
        truth_base = Path(f'data/{landmarks}/{target}')
        truth = utils.pload(str(truth_base / 'p_truth.pickle'))

        plt.figure()
        _relative_error = compute_relative_error(truth)
        for realisation in path.glob('REALISATION*'):
            p_means = []
            for p_mean_path in realisation.glob('P_mean_iter=*.pickle'):
                p_mean = utils.pload(str(p_mean_path))
                p_means.append(p_mean)

            relative_errors = list(map(_relative_error, p_means))
            plt.loglog(range(1, len(relative_errors) + 1), relative_errors)

        plt.xlabel(r'Iteration $k$', fontsize=20)
        plt.ylabel(r'Relative error', fontsize=20)
        plt.grid(linestyle='dotted')
        destination = str(path / f'../relative_error_{landmarks}_{ensemble_size}.pdf')
        plt.savefig(destination, bbox_inches='tight')
        plt.close()
        exit()
