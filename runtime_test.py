from loocv_assigmatcher_nov import getSILCALBIC, getloopcount, plot_bic_violin, get_clusassignments_from_LOOCV, \
    get_co_cluster_count, get_final_assignment, get_maxmatches, k_workup_mainfold, \
    match_assignments_to_final_assignments, get_aggregated_patterns, recode_iteration_assignments, \
    rand_score_comparison, collect_betas_for_corresponding_clus
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score as rand

from utils import ecdf, sdremoved_highest_val, max_min_val_check, rand_score_withnans

set = 7
subfold = 0
mainfold = 0

k=6

A2=get_clusassignments_from_LOOCV(set,mainfold,subfold)
A=A2[:,k-2,:]
co_cluster_count=get_co_cluster_count(A)

max_matches, max_locs = get_maxmatches(A, co_cluster_count, .8) # pair assignment patterns with high co-clustering
final_assignment = get_final_assignment(max_matches, 25) # unique assignment patterns from best pairs
print(len(final_assignment),'clusters in final assignment')
match_pct = match_assignments_to_final_assignments(A, final_assignment) # fit of all observations to these patterns
meet_fit_requirements = max_min_val_check(match_pct,50,25) # observations that can clearly be assigned to one of the patterns
print(len(meet_fit_requirements),'observations meet fit requirements for overlap with final assignments')
aggregated_patterns_arrays = get_aggregated_patterns(final_assignment,meet_fit_requirements,match_pct, A)

corresponding_cluster = recode_iteration_assignments(aggregated_patterns_arrays,k)



n_groups=corresponding_cluster.shape[1]
n_iterations=A.shape[0]
n_obs=A.shape[1]
checkclus=np.zeros(shape=[n_groups,n_iterations,n_obs])
consensus_label=np.full(n_obs,np.nan)
for group in range(n_groups):
    for i in range(n_iterations):
        checkclus[group,i,np.where(A[i,:]==corresponding_cluster[i,group])[0]]=1
for ppt in range(n_obs):
    crit=np.sum(checkclus[:,:,ppt],axis=1)
    consensus_label[ppt]=np.where(crit==np.max(crit))[0][0]


new_betas = collect_betas_for_corresponding_clus(corresponding_cluster,set, mainfold,subfold,k)
fig=plt.figure()
for n in range(len(new_betas)):

    plt.subplot(6,1,n+1)
    print(new_betas[n].shape[1],'beta vectors for cluster',n)
    plt.plot(new_betas[n])
#print(consensus_label)
#plt.scatter(np.arange(n_iterations),[rand_score_withnans(consensus_label,A[i,:]) for i in range(n_iterations)]);
plt.show()


#rand_all, rand_aftermatch = rand_score_comparison(A,consensus_label)

