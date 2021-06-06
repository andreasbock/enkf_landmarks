import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.utils as utils
from src.run_enkf import run_enkf_on_target

if __name__ == "__main__":

    # run the EnKF on all the manufactured solutions in the `data` directory
    target_paths = sorted(Path('data/LANDMARKS=50').glob('TARGET*'))
    destination_root = Path('RESULTS_REGULARISATION_EXPERIMENTS/')
    ensemble_size = 50
    max_iter = 50
    time_steps = 15
    regularisation_values = (0, 0.01, 0.1, 1, 10, 100)
    markers = ('.', '^', 's', '*', 'd', ' ')

    # run regularisation
    for target_path in target_paths:
        target_name = os.path.basename(str(target_path)).lstrip('TARGET_')
        for regularisation in regularisation_values:
            destination = destination_root / f"RESULT_{target_name}_regularisation={regularisation}/"
            run_enkf_on_target(str(target_path), str(destination), ensemble_size, time_steps, max_iter, regularisation)

        plt.figure()
        for result, marker in zip(sorted(destination_root.glob(f'RESULT_{target_name}_*')), markers):
            regularisation = re.search(f'regularisation=(.*)', str(result)).group(1)  # should have pickled this shouldn't we..
            misfits = [n.detach().numpy() for n in utils.pload(str(result / 'misfits.pickle'))]
            plt.semilogy(range(1, len(misfits) + 1), misfits, label=r'$\xi= ' + regularisation + '$',
                         marker=marker, markevery=3)

        handles, labels = plt.gca().get_legend_handles_labels()
        sorted_handles_and_labels = sorted(zip(handles, labels), key=lambda x: x[1])
        sorted_handles, sorted_labels = zip(*sorted_handles_and_labels)
        plt.legend(sorted_handles, sorted_labels, fontsize=12)

        plt.xlabel(r'Iteration $k$', fontsize=20)
        plt.ylabel(r'$\log E^k$', fontsize=20)
        plt.grid(linestyle='dotted')
        file_name = f'misfits_{target_name}.pdf'
        plt.savefig(str(destination_root / file_name), bbox_inches='tight')
        plt.close()
