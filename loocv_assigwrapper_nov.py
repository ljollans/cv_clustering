import statistics

from loocv_assigmatcher_nov import (
    get_k_from_bic,
    getloopcount,
    plot_bic_violin,
    get_clusassignments_from_LOOCV,
    get_co_cluster_count,
    get_final_assignment,
    get_maxmatches,
    k_workup_mainfold,
    match_assignments_to_final_assignments,
    get_aggregated_patterns,
    recode_iteration_assignments,
    rand_score_comparison,
    collect_betas_for_corresponding_clus,
    return_train_data,
)
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score as rand

from utils import ecdf, sdremoved_highest_val, max_min_val_check, rand_score_withnans

savedir = "/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_"
input_files_dir ="/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
null = 0

sets = ['Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct_s', 'Svcs_s', 'Tvct_Svcs_s',
            'Tvct_tvc_s',
            'Svcs_svc_s', 'Tvct_Svcs_tvc_svc_s', 'Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s',
            'Tct_Scs_s', 'Tct_tc_s', 'Scs_sc_s', 'Tct_Scs_tc_sc_s']

loopcount = getloopcount(savedir, null)
[
    best_sil,
    best_cal,
    best_bic,
    best_k_mf,
    best_k_sf,
    best_k_loocv,
    sil,
    cal,
    bic,
    pct_agreement_k,
] = get_k_from_bic(savedir, loopcount, null, sets)

rand_score_labelling_type=np.zeros(shape=[len(sets),2,4,4])

for s in range(len(sets)):
    for ctr in range(2):
        k = statistics.mode(best_k_mf[:, ctr, s]).astype(int)+2
        pkl_filename = (savedir + sets[s] + 'BETAS_fold0.pkl')
        with open(pkl_filename, "rb") as file:
            example_beta = pickle.load(file)

        if ctr==1:
            data_path = input_files_dir + "MDD__" + sets[s] + "_ctrl.csv"
        else:
            data_path = input_files_dir + "MDD__" + sets[s] + ".csv"

        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            X = np.array(list(reader)).astype(float)

        allbetas = np.full([4, 4, k, example_beta[0].shape[2]], np.nan)

        for mainfold in range(4):
            for subfold in range(4):
                # load the cluster assignments from all LOOCV iterations
                A2 = get_clusassignments_from_LOOCV(s, mainfold, subfold, sets, savedir)
                # select those for our chosen k
                A = A2[:, k - 2, :]
                # calculate co-clustering ratio for all observation pairs
                co_cluster_count = get_co_cluster_count(A)
                # choose the best matching pairs
                max_matches, max_match_locs = get_maxmatches(A, co_cluster_count,0.8)
                # use the best pairs to identify unique assignment patterns corresponding to clusters
                final_assignment = get_final_assignment(max_matches, 25)
                # calculate how well all observations fit to each of these top assignment patterns
                match_pct = match_assignments_to_final_assignments(A,final_assignment)
                # pick those observations that have a clear membership to one of the patterns
                meet_fit_requirements = max_min_val_check(match_pct, 50,25)
                # sort those observations into the matching pattern and collect their assignments from all iterations
                aggregated_patterns_arrays = get_aggregated_patterns(final_assignment, meet_fit_requirements, match_pct,A)
                # for each iteration identify what cluster corresponds to the new patterns
                corresponding_cluster = recode_iteration_assignments(aggregated_patterns_arrays, k)
                # aggregate the beta weights from all matched iteration clusters
                aggregated_betas, new_betas = collect_betas_for_corresponding_clus(
                    corresponding_cluster, s, mainfold, subfold, k
                )

                # calculate consensus cluster assignment for all observations (not using betas)
                n_groups = corresponding_cluster.shape[1]
                n_iterations = A.shape[0]
                n_obs = A.shape[1]
                checkclus = np.zeros(shape=[n_groups, n_iterations, n_obs])
                consensus_label = np.full(n_obs, np.nan)
                for group in range(n_groups):
                    for i in range(n_iterations):
                        checkclus[group, i, np.where(A[i, :] == corresponding_cluster[i, group])[0]] = 1
                for ppt in range(n_obs):
                    crit = np.sum(checkclus[:, :, ppt], axis=1)
                    consensus_label[ppt] = np.where(crit == np.max(crit))[0][0]

                # calculate consensus cluster assignment for all observations (using betas)
                x_fold=return_train_data(X,mainfold, subfold)
                y_fold=x_fold.dot(aggregated_betas)
                consensus_label_2 = np.full(n_obs, np.nan)
                for ppt in range(n_obs):
                    consensus_label_2[ppt]=np.where(y_fold[ppt,:]==np.max(y_fold[ppt,:]))[0]

                rand_score_labelling_type[s,ctr,mainfold,subfold]=rand_score_withnans(consensus_label,consensus_label_2)



# if ctr==0:
#    pkl_filename = (dir2 + 'FEB_' + sets[s] + 'BETA_AGGR' +  '.pkl')
# else:
#    pkl_filename = (dir2 + 'FEB_' + sets[s] + 'BETA_AGGR_ctrl'  + '.pkl')
# with open(pkl_filename, 'wb') as file:
#    pickle.dump(allbetas,file)
# print(pkl_filename)