import os
import glob
import numpy as np

from enkf import *
import utils


def run_enkf_on_target(data_dir, log_dir="./"):
    # where to dump results
    log_dir += f"EXAMPLE_{utils.date_string()}/"

    # 1) load initial ensemble and target from file
    pe = MomentumEnsemble.load(data_dir + "/pe.pickle")
    target = utils.pload(data_dir + "/target.pickle")

    # dump into log dir anyway
    pe.save(log_dir + "p_initial.pickle")
    utils.pdump(target, log_dir + "target.pickle")

    # 2) make a template to start from
    template_numpy = utils.circle(len(target))
    template = torch.tensor(template_numpy, dtype=torch_dtype, requires_grad=True)
    utils.plot_landmarks(file_name=log_dir + "template_and_target",
                         template=template,
                         target=target)

    # 3) perturb ensemble by either
    #   A: adding noise:      p = p_target + \eta, \eta noise
    #   B: multiplying noise: p = p_target + \eta, \eta noise
    #       Note: this can be at ensemble level or element level!
    low = -10
    high = 10
    w = [np.random.uniform(low, high) for _ in pe.ensemble]
    utils.pdump(w, log_dir + "weight_vector.pickle")
    pe.perturb(w)

    # 4) set up and run Kalman filter
    ke = EnsembleKalmanFilter(template, target, log_dir=log_dir)
    p = ke.run(pe, target)
    
    # dump the results
    p.save(log_dir + "p_result.pickle")

    # plot the results
    utils.plot_errors(log_dir + "errors.pickle", log_dir + "errors.pdf")
    utils.plot_consensus(log_dir + "consensus.pickle", log_dir + "errors.pdf")

    target_pickles = glob.glob(log_dir + '/PREDICTED_TARGET_iter=*.pickle')
    for target_pickle in target_pickles:
        file_name, _ = os.path.splitext(target_pickle)
        utils.plot_landmarks(qs=utils.pload(target_pickle),
                             template=template,
                             target=target,
                             file_name=file_name)


if __name__ == "__main__":
    # run the EnKF on all the manufactured solutions in the `data` directory
    for target_data in glob.glob('./data/TARGET*'):
        run_enkf_on_target(target_data)
