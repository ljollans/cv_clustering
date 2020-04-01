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

savedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'
null = 0

loopcount = getloopcount(savedir, null)
[best_sil, best_cal, best_bic, best_k_mf, best_k_sf, best_k_loocv, sil, cal, bic] = getSILCALBIC(savedir, loopcount,
                                                                                                 null)
pct_agreement_k = np.zeros(shape=[4, 4, 2, 24])
for mainfold in range(4):
    for subfold in range(4):
        for ctr in range(2):
            for s in range(24):
                pct_agreement_k[mainfold, subfold, ctr, s] = (
                    len(np.where(best_k_loocv[mainfold, subfold, :, ctr, s] == best_k_mf[mainfold, ctr, s])[0]))

# print(bestnclus_sf)


# if ctr==0:
#    pkl_filename = (dir2 + 'FEB_' + sets[s] + 'BETA_AGGR' +  '.pkl')
# else:
#    pkl_filename = (dir2 + 'FEB_' + sets[s] + 'BETA_AGGR_ctrl'  + '.pkl')
# with open(pkl_filename, 'wb') as file:
#    pickle.dump(allbetas,file)
# print(pkl_filename)
