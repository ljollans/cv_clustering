import csv
import numpy as np
import pickle
import matplotlib.pyplot as plt

from clusmetwrapper import cluster
from utils import get_gradient_change
from beta_aggregate import aggregate

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


# all_subfold_betas is a list of length n_cv
    # each entry has the shape nfeatures x nclusters


for current_set in [0]:
    for ctr in [0]:
        for mainfold in range(4):

            for k in [4]:
                allbetas=[]

                for subfold in range(4):
                    fold = (mainfold*4)+subfold

                    if ctr==1:
                        pkl_filename = (savedir + sets[current_set] + '_mod_ctrl_' + str(fold))
                    else:
                        pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
                    print(pkl_filename)
                    with open(pkl_filename, "rb") as file:
                        mod = pickle.load(file)

                allbetas.append(mod.allbetas[k+1])
                aggregated_betas, all_weighted_betas =  aggregate(allbetas)