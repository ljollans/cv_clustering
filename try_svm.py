import pickle
import csv
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils import select_trainset

pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/ALL/ALL_Tc_mod_0'
with open(pkl_filename, "rb") as file:
    mod = pickle.load(file)

mod.cluster_ensembles_new_classification()

