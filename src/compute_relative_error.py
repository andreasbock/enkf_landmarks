from collections import defaultdict
import numpy as np
import csv
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


def iter_sort(str: Path):
    return int(re.search('[0-9]+', str.name).group())


def default_dict():
    return defaultdict(default_dict)


if __name__ == "__main__":
    results = default_dict()
    # plot the misfits for each target/landmark/ensemble size together
    for path in Path('RESULTS_ENKF/').glob('LANDMARKS=*/TARGET*/ENSEMBLE_SIZE=*'):
        path_string = str(path)

        landmarks = re.search('LANDMARKS=([0-9]+)', path_string).group()
        ensemble_size = re.search('ENSEMBLE_SIZE=([0-9]+)', path_string).group()
        target = re.search('TARGET_(.)*/', path_string).group()[:-1]
        truth_base = Path(f'data/{landmarks}/{target}')
        truth = utils.pload(str(truth_base / 'p_truth.pickle'))
        _relative_error = compute_relative_error(truth)

        # plot relative error for each realisation
        plt.figure()
        sum_relative_error = 0.
        num_realisations = 0
        for realisation in path.glob('REALISATION*'):
            p_means = []
            for p_mean_path in sorted(realisation.glob('P_mean_iter=*.pickle'), key=iter_sort):
                p_mean = utils.pload(str(p_mean_path))
                p_means.append(p_mean)

            relative_errors = list(map(_relative_error, p_means))
            plt.loglog(range(1, len(relative_errors) + 1), relative_errors)
            sum_relative_error += relative_errors[-1]
            num_realisations += 1

        plt.xlabel(r'Iteration $k$', fontsize=20)
        plt.ylabel(r'Relative error', fontsize=20)
        plt.grid(linestyle='dotted')
        destination = str(path / f'../relative_error_{landmarks}_{ensemble_size}.pdf')
        plt.savefig(destination, bbox_inches='tight')
        plt.close()

        # write csv
        results[landmarks][target][ensemble_size] = sum_relative_error / num_realisations

    csv_path = Path('RESULTS_ENKF') / 'mean_relative_errors_{landmarks}.csv'
    header = ['10', '50', '100']  # ensemble size
    for landmark, data in results.items():
        with open(str(csv_path).format(landmarks=landmark), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for target, ensemble_sizes in data.items():
                sorted_es = sorted(ensemble_sizes, key=lambda x: int(re.search('[0-9]+', x).group()))
                writer.writerow([ensemble_sizes[es] for es in sorted_es])

