import csv
import numpy as np
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering')
from wrappers.calc_model_labels_proba import calc_model_labels_proba

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/MDD__"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_')
n=398
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

#calc_model_labels_proba(input_files_dir,savedir, n, 0, cv_assignment)
#calc_model_labels_proba(input_files_dir,savedir, n, 1, cv_assignment)

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/ALL/wspecsamp_"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/ALL/wspecsamp_')
n=740
with open((cv_assignment_dir + "CVassig740.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)
#calc_model_labels_proba(input_files_dir,savedir, n, 2, cv_assignment)

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/IXI/IXI_"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/IXI/IXI_')
n=544
with open((cv_assignment_dir + "CVassigIXI.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

calc_model_labels_proba(input_files_dir,savedir, n, 0, cv_assignment)