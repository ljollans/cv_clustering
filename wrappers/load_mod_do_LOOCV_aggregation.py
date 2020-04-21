import csv
import numpy as np
import pickle

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_')

# presets
sets = [
    "Tc",
    "Sc",
    "TSc",
    "Tc_tc",
    "Sc_sc",
    "TSc_tsc",
    "Tct_s",
    "Scs_s",
    "Tct_Scs_s",
    "Tct_tc_s",
    "Scs_sc_s",
    "Tct_Scs_tc_sc_s",
]

n_cv_folds = 4
n_ks = 8
# cv import
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

saveasthese = ["bic", "sil", "cal", "all_clus_labels", "auc", "f1", "betas"]
loadasthese = ["BIC", "SIL", "CAL", "allcluslabels", "AUC", "F1", "BETAS"]

for current_set in range(len(sets)):
    for ctr in range(2):
        # data import
        if ctr==1:
            data_path = input_files_dir + "MDD__" + sets[current_set] + "_ctrl.csv"
        else:
            data_path = input_files_dir + "MDD__" + sets[current_set] + ".csv"
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = np.array(list(reader)).astype(float)

        for mainfold in range(4):
            for subfold in range(4):
                fold = (mainfold*4)+subfold

                if ctr==1:
                    pkl_filename = (savedir + sets[current_set] + '_mod_ctrl_' + str(fold))
                else:
                    pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
                print(pkl_filename)
                with open(pkl_filename, "rb") as file:
                    mod = pickle.load(file)
                mod.cluster_ensembles()
                with open(pkl_filename, "wb") as file:
                    pickle.dump(mod,file)