import os
import re
import glob
import argparse
from pathlib import Path

import src.utils as utils
from src.enkf import EnsembleKalmanFilter
from src.manufacture_shape_data import make_normal_momentum


def run_enkf_on_target(ensemble_size,
                       time_steps,
                       max_iter,
                       regularisation,
                       source,
                       destination):
    # where to dump results
    utils.create_dir_from_path_if_not_exists(destination)

    # 1) load template and target from file
    template = utils.pload(source + '/template.pickle')
    target = utils.pload(source + '/target.pickle')

    # 2) set up filter
    ke = EnsembleKalmanFilter(template, target, log_dir=destination)

    # 3) create initial momentum
    mean = utils.pload(source + '/mean.pickle')
    std = utils.pload(source + '/std.pickle')
    p_ensemble_list = [make_normal_momentum(num_landmarks=len(target), mean=mean, std=std)
                       for _ in range(ensemble_size)]

    # dump stuff into log dir
    utils.pdump(p_ensemble_list, destination + '/pe_initial.pickle')
    utils.pdump(template, destination + '/template.pickle')
    utils.pdump(target, destination + '/target.pickle')

    # 4) run Kalman filter
    ke.logger.info(f'Loaded data from {source}.')
    ke.logger.info(f'Dumping files to {destination}.')
    ke.logger.info('Running EnKF...')

    pe_result = ke.run(p_ensemble_list,
                       regularisation=regularisation,
                       time_steps=time_steps,
                       max_iter=max_iter)

    # 5) dump the results
    utils.pdump(pe_result, destination + '/pe_result.pickle')

    ke.logger.info(f'Dumping results into {destination}...')

    template = utils.pload(destination + '/template.pickle')
    target = utils.pload(destination + '/target.pickle')

    ke.logger.info('Plotting errors...')
    utils.plot_misfits(destination + '/misfits.pickle', destination + '/misfits.pdf')

    ke.logger.info('Plotting consensus...')
    utils.plot_consensus(destination + '/consensus.pickle', destination + '/consensus.pdf')

    ke.logger.info('Plotting landmarks...')
    target_pickles = glob.glob(destination + '/Q_mean_iter=*.pickle')
    for target_pickle in target_pickles:
        k = int(re.search(r'\d+', os.path.basename(target_pickle)).group())
        file_name, _ = os.path.splitext(target_pickle)
        utils.plot_landmarks(qs=utils.pload(target_pickle),
                             template=template,
                             target=target,
                             file_name=file_name,
                             landmark_label='$F[P^{' + str(k) + '}]$')

    ke.logger.info('Plotting template and target...')
    utils.plot_landmarks(file_name=destination + '/template_and_target',
                         template=template,
                         target=target)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Runs the EnkF algorithm on the data directory.')
    parser.add_argument('--time_steps', type=int, default=15, help='Time steps for the forward map.')
    parser.add_argument('--max_iter', type=int, default=50, help='Number of EnKF iterations.')
    parser.add_argument('--regularisation', type=float, default=1, help='Regularisation parameter.')
    parser.add_argument('--realisations', type=int, default=20)
    parser.add_argument('--log_dir', type=str, default='results/')
    args = parser.parse_args()

    # mapping of landmarks to ensemble sizes we wish to simulate for
    ensemble_sizes = [10, 50, 150]

    # run the EnKF on all the manufactured solutions in the `data` directory
    for target_path in Path('data/').glob('LANDMARKS=*'):
        targets = Path(target_path).glob('TARGET*')
        for target in targets:
            source_directory = str(target)
            for ensemble_size in ensemble_sizes:
                for _ in range(args.realisations):
                    destination = Path(source_directory.replace('data', args.log_dir))
                    destination /= f'ENSEMBLE_SIZE={ensemble_size}/REALISATION_{utils.date_string()}/'
                    run_enkf_on_target(ensemble_size,
                                       args.time_steps,
                                       args.max_iter,
                                       args.regularisation,
                                       source_directory,
                                       str(destination))
