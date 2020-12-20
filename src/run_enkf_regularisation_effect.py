from pathlib import Path
from src.run_enkf import run_enkf_on_target


if __name__ == "__main__":

    # run the EnKF on all the manufactured solutions in the `data` directory
    target_paths = sorted(Path('data/LANDMARKS=50').glob('TARGET*'))
    destination = str(Path('./REGULARISATION_EXPERIMENTS2/'))
    ensemble_size = 50
    max_iter = 50
    time_steps = 15

    # run regularisation
    for target_path in target_paths:
        run_enkf_on_target(str(target_path), destination, ensemble_size, time_steps, max_iter, regularisation=0.01)
        run_enkf_on_target(str(target_path), destination, ensemble_size, time_steps, max_iter, regularisation=0.1)
        run_enkf_on_target(str(target_path), destination, ensemble_size, time_steps, max_iter, regularisation=1)
        run_enkf_on_target(str(target_path), destination, ensemble_size, time_steps, max_iter, regularisation=10)
        run_enkf_on_target(str(target_path), destination, ensemble_size, time_steps, max_iter, regularisation=100)
