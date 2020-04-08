import csv
import numpy as np
import time
from clusmetwrapper import cluster
import pickle
import sklearn
import matplotlib.pyplot as plt


seconds1 = time.time()

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"

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

mainfold = 0
subfold = 0
n_cv_folds = 4
n_ks = 8
current_set=8

# cv import
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

# data import
if current_set >= len(sets):
    data_path = input_files_dir + "MDD__" + sets[current_set - len(sets)] + "_ctrl.csv"
else:
    data_path = input_files_dir + "MDD__" + sets[current_set] + ".csv"
with open(data_path, "r") as f:
    reader = csv.reader(f, delimiter=",")
    data = np.array(list(reader)).astype(float)

# init mod
mod = cluster(data, n_ks, cv_assignment, mainfold, subfold, "full")

#pull in results
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_' + sets[current_set])
ctr = 0
fold = (mainfold * 4) + subfold
saveasthese = ["bic","sil","cal","all_clus_labels","auc","f1","betas"]
loadasthese = ["BIC","SIL","CAL","allcluslabels","AUC","F1","BETAS"]
for d in range(len(saveasthese)):
    if ctr == 1:
        fold_string = "ctrl_fold" + str(fold) + ".pkl"
    else:
        fold_string = "_fold" + str(fold) + ".pkl"
    pkl_filename = savedir + loadasthese[d] + fold_string
    with open(pkl_filename, "rb") as file:
        tmp = pickle.load(file)
    exec(saveasthese[d] + " = tmp")

mod.pull_in_saves(bic, sil, cal, all_clus_labels, auc, f1, betas)
#mod.plot_loocv()
#mod.aggregate_loocv()
mod.calc_consensus_matrix()
#mod.cophenetic_correlation()
mod.cluster_ensembles()
mod.consensus_labels_LJ()

plt.plot(np.nanmean(mod.randall_cluster_ensembles, axis=1))
plt.plot(np.nanmean(mod.randall_consensus_labels, axis=1))
plt.plot(np.nanmean(np.nanmean(mod.rand_all, axis=2),axis=1))
plt.legend(['cluster ensembles', 'my way', 'original'])
plt.show()

