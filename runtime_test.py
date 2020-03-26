import numpy as np
import matplotlib.pyplot as plt
from utils import many_co_cluster_match, contingency_matrix
import time
import pickle
import csv
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

use_subs = np.where(cv_assignment[:, 0] > 0)[0]
A_use = A[use_subs, :]
A_use = A_use[:, use_subs]
print(A_use.shape)
for i in range(len(use_subs)):
    where_nan = np.where(np.isnan(A_use[i, :]))[0]
    row_mean = np.nanmean(A_use[i, :])
    for nns in where_nan:
        A_use[i, nns] = row_mean

y_pred = KMeans(n_clusters=n_groups).fit_predict(A_use[:20,:].T)

fig=plt.figure()
for clus in range(n_groups):
    plt.subplot(2,3,clus+1)
    plt.imshow(A_use[:,np.where(y_pred==clus)[0]])
plt.show()
