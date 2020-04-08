import csv
import numpy as np
import time
from clusmetwrapper import cluster
import pickle
import sklearn
import matplotlib.pyplot as plt

from utils import rand_score_withnans

seconds1 = time.time()

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"

mainfold = 0
subfold = 0
n_cv_folds = 4
n_ks = 8
current_set=8

pkl_filename = 'tst.pkl'
with open(pkl_filename,"rb") as file:
    mod = pickle.load(file)
mod.plot_loocv()
mod.cophenetic_correlation()


