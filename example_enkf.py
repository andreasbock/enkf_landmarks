import os
import re
import glob
import numpy as np
import ensemble

from enkf import *
import utils


def run_enkf_on_target(data_dir, log_dir="./", use_manufactured_initial_momentum=True):
    ensemble_size = 10

    # where to dump results
    log_dir += f"EXAMPLE_{utils.date_string()}/"

    # 1) load target from file
    target = utils.pload(data_dir + "/target.pickle")
    num_landmarks = 10
    # 2) make a unit circle template to start from
    template_numpy = utils.circle(num_landmarks)
    template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)

    # 3) Get initial momentum
    if use_manufactured_initial_momentum:
        pe = MomentumEnsemble.load(data_dir + "/pe.pickle")
        low, high = -10, 10
        w = [np.random.uniform(low, high) for _ in pe.ensemble]
        utils.pdump(w, log_dir + "weight_vector.pickle")
        pe.perturb(w)
    else:
        pe = ensemble.ensemble_normal(num_landmarks, ensemble_size, scale=1.5)

    # dump stuff into log dir
    pe.save(log_dir + "p_initial.pickle")
    utils.pdump(template, log_dir + "template.pickle")
    utils.pdump(target, log_dir + "target.pickle")

    # 4) set up and run Kalman filter
    ke = EnsembleKalmanFilter(template, target, log_dir=log_dir)
    p = ke.run(pe, target)

    # 5) dump or plot the results
    p.save(log_dir + "p_result.pickle")

    return log_dir


def dump_results(log_dir):
    print(f"Dumping results into {log_dir}...")
    log_dir += '/'
    template = utils.pload(log_dir + "template.pickle")
    target = utils.pload(log_dir + "target.pickle")

    print("\t plotting errors...")
    utils.plot_errors(log_dir + "errors.pickle", log_dir + "errors.pdf")

    print("\t plotting consensus...")
    utils.plot_consensus(log_dir + "consensus.pickle", log_dir + "consensus.pdf")

    print("\t plotting weights...")
    w = utils.pload(log_dir + "weight_vector.pickle")
    with open(log_dir + "weight_vector_norm.log", "w+") as file:
        file.write(f"Norm of w: {np.linalg.norm(w)}")

    print("\t plotting landmarks...")
    target_pickles = glob.glob(log_dir + '/PREDICTED_TARGET_iter=*.pickle')
    for target_pickle in target_pickles:
        k = int(re.search(r'\d+', os.path.basename(target_pickle)).group())
        file_name, _ = os.path.splitext(target_pickle)
        utils.plot_landmarks(qs=utils.pload(target_pickle),
                             template=template,
                             target=target,
                             file_name=file_name,
                             landmark_label="$F[P^{" + str(k) + "}]$")

    print("\t plotting template and target...")
    utils.plot_landmarks(file_name=log_dir + "template_and_target",
                         template=template,
                         target=target)


if __name__ == "__main__":
    # run the EnKF on all the manufactured solutions in the `data` directory
    for target_data in glob.glob('./data/TARGET*'):
        log_dir = run_enkf_on_target(target_data, use_manufactured_initial_momentum=False)
        dump_results(log_dir)
        break
