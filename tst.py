import pickle
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')

#input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/null/MDDnull/MDD__'
#modstr = '_mod_null_'

input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'
modstr = '_mod_ctrl_'

sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s", "Tct_Scs_tc_sc_s"]
n_cv_folds = 4
n = 398
n_k = 8


s=0
fold=0

filestr=(input_filedir + sets[s] + modstr + str(fold))
with open(filestr, "rb") as f:
    mod = pickle.load(f)

mod.cluster_ensembles_new_classification()
mod.sf_class_probas()


print(np.nanmean(mod.testset_prob,axis=0))