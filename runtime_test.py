import csv
import numpy as np
import time
from clusmetwrapper import cluster
import pickle
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from loocv_assigmatcher_nov import collect_betas_for_corresponding_clus, sort_into_clusters_argmax_ecdf
from utils import rand_score_withnans, ecdf

seconds1 = time.time()

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"

mainfold = 0
subfold = 0
n_cv_folds = 4
n_ks = 8
current_set=8

pkl_filename = 'tst.pkl'
with open(pkl_filename,"rb") as file:
    mod = pickle.load(file)
mod.plot_loocv()
mod.cophenetic_correlation()

# are assignments based on betas in line with consensus_labels
data = mod.data[mod.train_index_sub,:]
for k in range(mod.nk):
    betas = mod.betas[mod.train_index_sub,k,:k+2,:]
    assignments = mod.iteration_assignments[k]

    aggregated_betas, new_betas_array = collect_betas_for_corresponding_clus(assignments, betas)

    argmax_assig, all_ys, Y = sort_into_clusters_argmax_ecdf(data, aggregated_betas)

    argmax_assig = np.full(Y.shape[0],np.nan)
    for ppt in range(Y.shape[0]):
        crit=all_ys[ppt,:]
        if np.max(crit)>.8:
            argmax_assig[ppt]=np.where(crit==np.max(crit))[0][0]

    print(rand_score_withnans(argmax_assig, mod.cluster_ensembles_labels[:,k]))

