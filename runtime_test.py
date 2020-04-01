from loocv_assigmatcher_nov import getSILCALBIC, getloopcount, plot_bic_violin, get_clusassignments_from_LOOCV, \
    get_co_cluster_count, get_final_assignment, get_maxmatches, k_workup_mainfold, \
    match_assignments_to_final_assignments
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

#k_workup_mainfold(mainfold,set)

A2=get_clusassignments_from_LOOCV(set,mainfold,subfold)
A=A2[:,4,:]
co_cluster_count=get_co_cluster_count(A)

max_matches, max_locs = get_maxmatches(A, co_cluster_count, .8) # pair assignment patterns with high co-clustering
final_assignment = get_final_assignment(max_matches, 25) # unique assignment patterns from best pairs
match_pct = match_assignments_to_final_assignments(A, final_assignment) # fit of all observations to these patterns
meet_fit_requirements = max_min_val_check(match_pct,50,25) # observations that can clearly be assigned to one of the patterns

aggregated_patterns=[]
for p in range(len(final_assignment)):
    aggregated_patterns.append([])

for ppt in meet_fit_requirements:
    ppt_fit=np.where(match_pct[ppt,:]>50)[0]
    aggregated_patterns[ppt_fit[0]].append(A[:,ppt])

print([len(aggregated_patterns[i]) for i in range(len(final_assignment))])




