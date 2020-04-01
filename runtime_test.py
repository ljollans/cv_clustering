from loocv_assigmatcher_nov import getSILCALBIC, getloopcount, plot_bic_violin, get_clusassignments_from_LOOCV, \
    get_co_cluster_count, get_final_assignment, get_maxmatches, k_workup_mainfold, \
    match_assignments_to_final_assignments
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import ecdf, distinctiveness_highest_val

set = 7
subfold = 0
mainfold = 0

#k_workup_mainfold(mainfold,set)

A2=get_clusassignments_from_LOOCV(set,mainfold,subfold)
A=A2[:,4,:]
co_cluster_count=get_co_cluster_count(A)

maxmatches, findmax = get_maxmatches(A, co_cluster_count, .8)

print(maxmatches.shape)

final_assignment = get_final_assignment(maxmatches, 25)
print(len(final_assignment))

match_pct = match_assignments_to_final_assignments(A, final_assignment)
max_fit=np.max(match_pct,axis=1)
clear_max=distinctiveness_highest_val(match_pct)
print(np.min(max_fit[clear_max]))
plt.hist(max_fit[clear_max]); plt.show()
# only use pct >=50 overlap
