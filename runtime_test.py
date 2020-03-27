import numpy as np
from utils import many_co_cluster_match, contingency_matrix
import pickle
import csv
import matplotlib.pyplot as plt

cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

n_groups = 6

pkl_filename = "/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_svc_sallcluslabels_fold0.pkl"
with open(pkl_filename, "rb") as file:
    A = pickle.load(file)
A = np.squeeze(A[:, n_groups - 2, :])

co_cluster_count,  maxmatches, equal_check = many_co_cluster_match(A, n_groups)

print(equal_check)