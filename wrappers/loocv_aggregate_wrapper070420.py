import csv
import numpy as np
from cv_clustering.clusmetwrapper import cluster
import pickle
from cv_clustering.loocv_assigmatcher_nov import pull_from_mod2

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
path2use_1='/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'
path2use_2='/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'

sets = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
         'Scs_sc_s', 'Tct_Scs_tc_sc_s']

# cv import
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

for current_set in [0]:
    savedir = (path2use_1 + sets[current_set])
    for ctr in range(2):
        # data import
        if ctr==1:
            data_path = input_files_dir + "MDD__" + sets[current_set - len(sets)] + "_ctrl.csv"
        else:
            data_path = input_files_dir + "MDD__" + sets[current_set] + ".csv"
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = np.array(list(reader)).astype(float)

        s_PAC=[]
        s_RAND=[]
        s_COPH=[]
        s_BetaAggr=[]
        s_BetaPreAggr=[]
        s_ConLab=[]


        for mainfold in range(4):
            for subfold in range(4):
                fold = (mainfold * 4) + subfold
                mod = cluster(data, 8, cv_assignment, mainfold, subfold, "full")
                saveasthese = ["bic", "sil", "cal", "all_clus_labels", "auc", "f1", "betas"]
                loadasthese = ["BIC", "SIL", "CAL", "allcluslabels", "AUC", "F1", "BETAS"]
                for d in range(len(saveasthese)):
                    if ctr == 1:
                        fold_string = "_ctrl_fold" + str(fold) + ".pkl"
                    else:
                        fold_string = "_fold" + str(fold) + ".pkl"
                    pkl_filename = savedir  + loadasthese[d] +  fold_string
                    with open(pkl_filename, "rb") as file:
                        tmp = pickle.load(file)
                    exec(saveasthese[d] + " = tmp")

                mod.pull_in_saves(bic, sil, cal, all_clus_labels, auc, f1, betas)

                #mod.aggregate_loocv()
                mod.calc_consensus_matrix()
                #mod.get_pac()
                #mod.calc_rand_all()
                mod.cluster_ensembles()
                mod.cluster_ensembles_match_betas()
                mod.cophenetic_correlation()

                tmp_coph, tmp_aggrbetas, tmp_preaggrbetas, tmp_conlab = pull_from_mod2(mod)
                s_COPH.append(tmp_coph)
                s_BetaAggr.append(tmp_aggrbetas)
                s_BetaPreAggr.append(tmp_preaggrbetas)
                s_ConLab.append(tmp_conlab)

        if ctr==1:
            ss=((path2use_2 + sets[current_set] + '_ctrl_'))
        else:
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