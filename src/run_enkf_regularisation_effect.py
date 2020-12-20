import os
import re
import glob
import numpy as np
from pathlib import Path

import src.utils as utils
from src.enkf import EnsembleKalmanFilter
from src.manufacture_shape_data import make_normal_momentum


def run_enkf_on_target(source_directory,
                       destination,
                       ensemble_size,
                       regularisation):
    source_directory_string = str(source_directory)

    # where to dump results
    target_name = os.path.basename(source_directory_string).lstrip('TARGET_')
    destination /= f"RESULT_{target_name}_regularisation={regularisation}_{utils.date_string()}/"
    destination_string = str(destination)

    utils.create_dir_from_path_if_not_exists(destination_string)

    # 1) load template and target from file
    template = utils.pload(source_directory_string + "/template.pickle")
    target = utils.pload(source_directory_string + "/target.pickle")

    # 2) set up filter
    ke = EnsembleKalmanFilter(template, target, log_dir=destination_string)

    # 3) load initial momentum
    mean = utils.pload(source_directory_string + '/mean.pickle')
    std = utils.pload(source_directory_string + '/std.pickle')
    p_ensemble_list = [make_normal_momentum(num_landmarks=len(target), mean=mean, std=std)
                       for _ in range(ensemble_size)]

    # dump stuff into log dir
    utils.pdump(p_ensemble_list, destination_string + "/p_ensemble_truth.pickle")
    utils.pdump(template, destination_string + "/template.pickle")
    utils.pdump(target, destination_string + "/target.pickle")

    # 4) run Kalman filter
    ke.logger.info(f"Loaded data from {source_directory_string}.")
    ke.logger.info(f"Dumping files to {destination_string}.")
    ke.logger.info("Running EnKF...")

    pe_result = ke.run(p_ensemble_list, regularisation=regularisation)

    # 5) dump the results
    utils.pdump(pe_result, destination_string + "/pe_result.pickle")

    # if we have used a perturbed version of the true initial ensemble
    # as our initial ensemble then dump the perturbation vector
    w_vector_path = source_directory_string + "/weight_vector.pickle"
    if os.path.exists(w_vector_path):
        w = utils.pload(w_vector_path)
        distance_from_unit = np.linalg.norm(w - np.eye(len(w)))
        ke.logger.info(f"Weight vector used has norm: {np.linalg.norm(w)}, l^2 distance from the unit vector: "
                       f"{distance_from_unit}.")

    ke.logger.info(f"Dumping results into {destination_string}...")

    template = utils.pload(destination_string + "/template.pickle")
    target = utils.pload(destination_string + "/target.pickle")

    ke.logger.info("Plotting errors...")
    utils.plot_misfits(destination_string + "/misfits.pickle", destination_string + "/misfits.pdf")

    ke.logger.info("Plotting consensus...")
    utils.plot_consensus(destination_string + "/consensus.pickle", destination_string + "/consensus.pdf")

    ke.logger.info("Plotting landmarks...")
    target_pickles = glob.glob(destination_string + '/Q_mean_iter=*.pickle')
    for target_pickle in target_pickles:
        k = int(re.search(r'\d+', os.path.basename(target_pickle)).group())
        file_name, _ = os.path.splitext(target_pickle)
        utils.plot_landmarks(qs=utils.pload(target_pickle),
                             template=template,
                             target=target,
                             file_name=file_name,
                             landmark_label="$F[P^{" + str(k) + "}]$")

    ke.logger.info("Plotting template and target...")
    utils.plot_landmarks(file_name=destination_string + "/template_and_target",
                         template=template,
                         target=target)


if __name__ == "__main__":

    # run the EnKF on all the manufactured solutions in the `data` directory
    target_paths = sorted(Path('data/LANDMARKS=50').glob('TARGET*'))
    destination = Path('./REGULARISATION_EXPERIMENTS/')
    ensemble_size = 50

    # run regularisation
    for target_path in target_paths:
        run_enkf_on_target(target_path, destination, ensemble_size, regularisation=0.01)
        run_enkf_on_target(target_path, destination, ensemble_size, regularisation=0.1)
        run_enkf_on_target(target_path, destination, ensemble_size, regularisation=1)
        run_enkf_on_target(target_path, destination, ensemble_size, regularisation=10)
        run_enkf_on_target(target_path, destination, ensemble_size, regularisation=100)
