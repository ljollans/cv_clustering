import csv
import numpy as np
import sys
import time
import pickle
import sklearn

from clusmetwrapper import cluster
from utils import identify_set_and_fold, rand_score_withnans
import sys

sys.path.append(
    "/Users/lee_jollans/PycharmProjects/Cluster_Ensembles/src/Cluster_Ensembles"
)
import Cluster_Ensembles as CE

seconds1 = time.time()

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
save_dir = "/Users/lee_jollans/Projects/clustering_pilot/ALL/"
save_str = "ALL_"

# presets
sets = [
    "Tc",
    "Sc",
    "TSc",
    "Tc_tc",
    "Sc_sc",
    "TSc_tsc",
    "Tct_s",
    "Scs_s",
    "Tct_Scs_s",
    "Tct_tc_s",
    "Scs_sc_s",
    "Tct_Scs_tc_sc_s",
]

n_cv_folds = 4
n_ks = 8
with open((cv_assignment_dir + "CVassig740.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

# settings
current_proc = 0
current_set, current_fold = identify_set_and_fold(current_proc, n_cv_folds)
mainfolds_subfolds = [
    np.repeat(np.arange(n_cv_folds), n_cv_folds),
    np.tile(np.arange(n_cv_folds), n_cv_folds),
]

# data import
data_path = input_files_dir + "ALL_" + sets[current_set] + ".csv"

with open(data_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    data = np.array(list(reader)).astype(float)

mod = cluster(
    data,
    n_ks,
    cv_assignment,
    mainfolds_subfolds[0][current_fold].astype(int),
    mainfolds_subfolds[1][current_fold].astype(int),
    "full",
)

mod.run()

pkl_filename = save_dir + save_str + sets[current_set] + '_mod_' + str(current_fold)
with open(pkl_filename,"wb") as file:
    pickle.dump(mod, file)