import statistics

from loocv_assigmatcher_nov import (
    get_k_from_bic,
    getloopcount,
    get_clusassignments_from_LOOCV,
    get_co_cluster_count,
    get_final_assignment,
    get_maxmatches,
    match_assignments_to_final_assignments,
    get_aggregated_patterns,
    recode_iteration_assignments,
    collect_betas_for_corresponding_clus,
)
import pickle
import csv
import numpy as np

from utils import max_min_val_check

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

        allbetas = np.full([4, 4, k+3, example_beta[0].shape[2]], np.nan)

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
                    corresponding_cluster, s, mainfold, subfold, k, sets
                )

                allbetas[mainfold,subfold,:aggregated_betas.shape[1],:]=aggregated_betas.T

        if ctr==0:
            pkl_filename = (savedir + sets[s] + 'BETA_AGGR' +  '.pkl')
        else:
            pkl_filename = (savedir + sets[s] + 'BETA_AGGR_ctrl'  + '.pkl')
        with open(pkl_filename, 'wb') as file:
            pickle.dump(allbetas,file)
        print(pkl_filename)