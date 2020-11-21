import os
import re
import glob
import numpy as np
from pathlib import Path

from enkf_landmarks.enkf import *


def run_enkf_on_target(data_dir,
                       log_dir='../',
                       use_regularisation=True):

    # where to dump results
    regularised = str(use_regularisation)
    target_name = os.path.basename(data_dir).lstrip('TARGET_')
    log_dir += f"RESULT_{target_name}_regularised={regularised}_{utils.date_string()}/"

    # 1) load template and target from file
    template = utils.pload(data_dir + "/template.pickle")
    target = utils.pload(data_dir + "/target.pickle")

    # 2) load initial momentum
    pe = MomentumEnsemble.load(data_dir + "/pe_initial.pickle")

    # dump stuff into log dir
    pe.save(log_dir + "pe_initial.pickle")
    utils.pdump(template, log_dir + "template.pickle")
    utils.pdump(target, log_dir + "target.pickle")

    # 4) set up and run Kalman filter
    ke = EnsembleKalmanFilter(template, target, log_dir=log_dir)
    pe_result = ke.run(pe, target, with_regularisation=use_regularisation)

    # 5) dump the results
    pe_result.save(log_dir + "pe_result.pickle")

    # if we have used a perturbed version of the true initial ensemble
    # as our initial ensemble then dump the perturbation vector
    w_vector_path = data_dir + "/weight_vector.pickle"
    if os.path.exists(w_vector_path):
        w = utils.pload(w_vector_path)
        distance_from_unit = np.linalg.norm(w - np.eye(len(w)))
        ke.logger.info(f"Weight vector used has norm: {np.linalg.norm(w)}, l^2 distance from the unit vector: "
                       f"{distance_from_unit}.")

    ke.logger.info(f"Dumping results into {log_dir}...")
    log_dir += '/'
    template = utils.pload(log_dir + "template.pickle")
    target = utils.pload(log_dir + "target.pickle")

    ke.logger.info("Plotting errors...")
    utils.plot_errors(log_dir + "errors.pickle", log_dir + "errors.pdf")

    ke.logger.info("Plotting consensus...")
    utils.plot_consensus(log_dir + "consensus.pickle", log_dir + "consensus.pdf")

    ke.logger.info("Plotting landmarks...")
    target_pickles = glob.glob(log_dir + '/PREDICTED_TARGET_iter=*.pickle')
    for target_pickle in target_pickles:
        k = int(re.search(r'\d+', os.path.basename(target_pickle)).group())
        file_name, _ = os.path.splitext(target_pickle)
        utils.plot_landmarks(qs=utils.pload(target_pickle),
                             template=template,
                             target=target,
                             file_name=file_name,
                             landmark_label="$F[P^{" + str(k) + "}]$")

    ke.logger.info("Plotting template and target...")
    utils.plot_landmarks(file_name=log_dir + "template_and_target",
                         template=template,
                         target=target)


if __name__ == "__main__":
    # run the EnKF on all the manufactured solutions in the `data` directory
    target_paths = sorted(Path('../data/').glob('TARGET*'))

    for target_path in target_paths:
        run_enkf_on_target(str(target_path))
        run_enkf_on_target(str(target_path), use_regularisation=False)
