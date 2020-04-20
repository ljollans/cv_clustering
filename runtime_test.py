import csv
import numpy as np
import pickle

from clusmetwrapper import cluster

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



for current_set in [1,2,3,4,5,6,7,8,9,10,11]:
    for ctr in range(2):

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