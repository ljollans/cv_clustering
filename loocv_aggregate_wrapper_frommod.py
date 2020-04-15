import csv
import numpy as np
import time
from clusmetwrapper import cluster
import pickle
from loocv_assigmatcher_nov import pull_from_mod2

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
path2use_1='/Users/lee_jollans/Projects/clustering_pilot/ALL/ALL_'
path2use_2='/Users/lee_jollans/Projects/clustering_pilot/ALL/ALL_'

sets = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
         'Scs_sc_s', 'Tct_Scs_tc_sc_s']

# cv import
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

for current_set in [1]:
    print(sets[current_set])
    savedir = (path2use_1 + sets[current_set])

    s_COPH=[]
    s_BetaAggr=[]
    s_BetaPreAggr=[]
    s_ConLab=[]
    for fold in range(16):
        print(fold)
        pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/ALL/ALL_' + sets[current_set] + '_mod_' + str(fold)
        with open(pkl_filename, "rb") as file:
            mod = pickle.load(file)
        #mod.aggregate_loocv()
        mod.calc_consensus_matrix()
        #mod.get_pac()
        #mod.calc_rand_all()
        mod.cluster_ensembles()

        with open(pkl_filename, "wb") as file:
            pickle.dump(mod,file)

        mod.cluster_ensembles_match_betas()
        mod.cophenetic_correlation()

        with open(pkl_filename, "wb") as file:
            pickle.dump(mod,file)

        tmp_coph, tmp_aggrbetas, tmp_preaggrbetas, tmp_conlab = pull_from_mod2(mod)
        s_COPH.append(tmp_coph)
        s_BetaAggr.append(tmp_aggrbetas)
        s_BetaPreAggr.append(tmp_preaggrbetas)
        s_ConLab.append(tmp_conlab)

    ss=((path2use_2 + sets[current_set] + '_'))
    dumpthese = [
        "COPH",
        "BetaAggr",
        "BetaPreAggr",
        "ConLab",
    ]
    for d in dumpthese:
        pkl_filename = ss + '_' + d + '.pkl'
        with open(pkl_filename, "wb") as file:
            eval("pickle.dump(s_" + d + ", file)")