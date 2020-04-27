#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 13:50:59 2020

@author: lee_jollans
"""

import csv
import numpy as np
import sys
import time
from cv_clustering.clusmetwrapper import cluster
from cv_clustering.utils import identify_set_and_fold

seconds1 = time.time()

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
save_dir = "/Users/lee_jollans/Projects/clustering_pilot/tstsave/"
save_str = sys.argv[2]

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
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

# settings
current_proc = int(sys.argv[1])
current_set, current_fold = identify_set_and_fold(current_proc, n_cv_folds)
mainfolds_subfolds = [
    np.repeat(np.arange(n_cv_folds), n_cv_folds),
    np.tile(np.arange(n_cv_folds), n_cv_folds),
]

# data import
if current_set >= len(sets):
    data_path = input_files_dir + "MDD__" + sets[current_set - len(sets)] + "_ctrl.csv"
else:
    data_path = input_files_dir + "MDD__" + sets[current_set] + ".csv"

with open(data_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    data = np.array(list(reader)).astype(float)


mod = cluster(
    data,
    n_ks,
    cv_assignment,
    mainfolds_subfolds[0][current_fold].astype(int),
    mainfolds_subfolds[1][current_fold].astype(int),
    sys.argv[3],
)
mod.run()

# save
dumpthese = [
    "bic",
    "sil",
    "cal",
    "all_clus_labels",
    "auc",
    "f1",
    "betas",
    "n_per_classification",
]
for d in dumpthese:
    if current_set >= len(sets):
        fold_string = "ctrl_fold" + str(current_fold) + ".pkl"
    else:
        fold_string = "_fold" + str(current_fold) + ".pkl"
    pkl_filename = save_dir + save_str + sets[current_set] + d + fold_string
    with open(pkl_filename, "wb") as file:
        eval("pickle.dump(mod." + d + ", file)")

seconds = time.time()
print("Seconds since epoch =" + str(seconds - seconds1))
