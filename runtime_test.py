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
    n_clus_retrieval_chk)
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score as rand

from utils import ecdf, sdremoved_highest_val, max_min_val_check, rand_score_withnans

set = 5
subfold = 1
mainfold = 0

k = 6

sets = ['Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct_s', 'Svcs_s', 'Tvct_Svcs_s', 'Tvct_tvc_s',
            'Svcs_svc_s', 'Tvct_Svcs_tvc_svc_s']
sets_null = ['Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct', 'Svcs', 'Tvct_Svcs', 'Tvct_tvc',
            'Svcs_svc', 'Tvct_Svcs_tvc_svc']
savedir_null='/Users/lee_jollans/Projects/clustering_pilot/null/JAN1_'
savedir='/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'

A2 = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets, savedir, 0)
A = A2

A2 = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets_null, savedir_null, 1)
A_null = A2

ex1=getloopcount(savedir, 0)
ex1_null=getloopcount(savedir_null, 1)
[bestSIL, bestCAL, bestBIC,bestnclus_mf, bestnclus_sf, best_k_loocv, SIL, CAL, BIC, pct_agreement_k] = get_k_from_bic(savedir, ex1, 0, sets)
[bestSIL_null, bestCAL_null,bestBIC_null, bestnclus_mf_null, bestnclus_sf_null,best_k_loocv_null, SIL_null, CAL_null, BIC_null, pct_agreement_k_null] = \
    get_k_from_bic(savedir_null, ex1_null, 1, sets_null)

cr, pac, plateau_cr = n_clus_retrieval_chk(A, .7, 25)
cr_null, pac_null, plateau_cr_null = n_clus_retrieval_chk(A_null, .7, 25)

fig = plt.figure();
plt.subplot(2,2,1); plt.title('actual bic');  plot_bic_violin(BIC, mainfold, subfold, 0, set);
plt.subplot(2,2,2); plt.title('null bic');  plot_bic_violin(BIC_null, mainfold, subfold, 0, set);
plt.subplot(2,2,3); plt.title('saturation cluster')
plt.plot(cr); plt.plot(cr_null); plt.legend(['actual','null'])
plt.subplot(2,2,4); plt.title('pac')
plt.plot(pac_null-pac); plt.legend(['null-actual'])
plt.show()


#match_pct = match_assignments_to_final_assignments(A, final_assignment)  # fit of all observations to these patterns
#meet_fit_requirements = max_min_val_check(match_pct, 50, 25)  # observations that can clearly be assigned to one of the patterns
#print(len(meet_fit_requirements),"observations meet fit requirements for overlap with final assignments",)
#aggregated_patterns_arrays = get_aggregated_patterns(final_assignment, meet_fit_requirements, match_pct, A)#

#corresponding_cluster = recode_iteration_assignments(aggregated_patterns_arrays, k)


#n_groups = corresponding_cluster.shape[1]
#n_iterations = A.shape[0]
#n_obs = A.shape[1]
#checkclus = np.zeros(shape=[n_groups, n_iterations, n_obs])
#consensus_label = np.full(n_obs, np.nan)
#for group in range(n_groups):
#    for i in range(n_iterations):
#        checkclus[group, i, np.where(A[i, :] == corresponding_cluster[i, group])[0]] = 1
#for ppt in range(n_obs):
#    crit = np.sum(checkclus[:, :, ppt], axis=1)
#    consensus_label[ppt] = np.where(crit == np.max(crit))[0][0]


#aggregated_betas, new_betas = collect_betas_for_corresponding_clus(
#    corresponding_cluster, set, mainfold, subfold, k
#)

#fig = plt.figure()
#plt.plot(aggregated_betas)
#plt.show()


# rand_all, rand_aftermatch = rand_score_comparison(A,consensus_label)
