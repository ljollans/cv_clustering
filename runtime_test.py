import numpy as np
import matplotlib.pyplot as plt
from utils import many_co_cluster_match, contingency_matrix
import time
import pickle
import csv
import pandas as pd
from sklearn.cluster import KMeans

cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

n_groups = 6

pkl_filename = "/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_svc_sallcluslabels_fold0.pkl"
with open(pkl_filename, "rb") as file:
    A = pickle.load(file)
A = np.squeeze(A[:, n_groups - 2, :])

filename='/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/sample_descriptors.csv'
sample_descriptors = pd.read_csv(filename)
sample_descriptors.describe()

scanstudy=sample_descriptors['scanstudy'].to_numpy()

#assignments_to_use=np.where(cv_assignment[:,0]>0)[0]
co_cluster_count, overlap = many_co_cluster_match(A, n_groups)#

fig=plt.figure()
plt.subplot(3,3,1); plt.imshow(A); plt.colorbar()
plt.subplot(3,3,2); plt.imshow(co_cluster_count); plt.colorbar()
for i in range(3):
    xx=co_cluster_count[np.where(scanstudy==i+1)[0],:]
    xx=xx[:,np.where(scanstudy==i+1)[0]]
    plt.subplot(3,3,4+i); plt.imshow(xx); plt.colorbar()
for i in range(3):
    xx=co_cluster_count[np.where(scanstudy==i+1)[0],:]
    xx=xx[:,np.where(scanstudy!=i+1)[0]]
    plt.subplot(3,3,7+i); plt.imshow(xx); plt.colorbar()
plt.show()