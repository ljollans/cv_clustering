import csv
import numpy as np
import sys
import time
from clusmetwrapper import cluster
from loocv_assigmatcher_nov import get_clusassignments_from_LOOCV
from utils import identify_set_and_fold
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import Cluster_Ensembles as CE

seconds1 = time.time()

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"

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

A2=get_clusassignments_from_LOOCV(0, 0, 0,sets,'/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_',0,0)

consensus_clustering_labels = CE.cluster_ensembles(A2[:,4,:], verbose = True, N_clusters_max = 50)