from loocv_assigmatcher_nov import getSILCALBIC, getloopcount, plot_bic_violin, get_clusassignments_from_LOOCV, \
    get_co_cluster_count, get_final_assignment, get_maxmatches, k_workup_mainfold, \
    match_assignments_to_final_assignments, get_aggregated_patterns, recode_iteration_assignments
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import ecdf, sdremoved_highest_val, max_min_val_check

set = 1
subfold = 0
mainfold = 0

k=6

A2=get_clusassignments_from_LOOCV(set,mainfold,subfold)
A=A2[:,k-2,:]
co_cluster_count=get_co_cluster_count(A)

max_matches, max_locs = get_maxmatches(A, co_cluster_count, .8) # pair assignment patterns with high co-clustering
final_assignment = get_final_assignment(max_matches, 25) # unique assignment patterns from best pairs
match_pct = match_assignments_to_final_assignments(A, final_assignment) # fit of all observations to these patterns
meet_fit_requirements = max_min_val_check(match_pct,50,25) # observations that can clearly be assigned to one of the patterns
print(len(meet_fit_requirements))
aggregated_patterns_arrays = get_aggregated_patterns(final_assignment,meet_fit_requirements,match_pct, A)

corresponding_cluster = recode_iteration_assignments(aggregated_patterns_arrays,k)

fig=plt.figure()
for group in range(corresponding_cluster.shape[1]):
    plt.subplot(2,3,group+1)
    checkclus=np.zeros(shape=[A.shape[0],A.shape[1]])
    for i in range(A.shape[0]):
        checkclus[i,np.where(A[i,:]==corresponding_cluster[i,group])[0]]=1
    plt.imshow(checkclus);
plt.show()






