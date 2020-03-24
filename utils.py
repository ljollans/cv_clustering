import numpy as np


def identify_set_and_fold(current_proc, n_cv_folds):
    folds_per_run = n_cv_folds * n_cv_folds
    current_set = np.floor_divide(current_proc, folds_per_run)
    current_set_count_start = current_set * folds_per_run
    current_fold = current_proc - current_set_count_start
    return current_set, current_fold
