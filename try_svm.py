import pickle
import csv
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils import select_trainset
from clusmetwrapper import kcluster

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
path2use_1='/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'
path2use_2='/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'

sets = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
         'Scs_sc_s', 'Tct_Scs_tc_sc_s']


for current_set in range(len(sets)):
    for ctr in range(2):
        for mainfold in range(4):
            for subfold in range(4):
                fold = (mainfold*4)+subfold
                if ctr == 1:
                    ss = ((path2use_2 + sets[current_set] + '_ctrl_'))
                else:
                    ss = ((path2use_2 + sets[current_set] + '_'))
                pkl_filename = ss + '_' + d + '.pkl'
                with open(pkl_filename, "rb") as file:
                    mod = pickle.load(file)



pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/ALL/ALL_Tc_mod_1'
with open(pkl_filename, "rb") as file:
    mod = pickle.load(file)

