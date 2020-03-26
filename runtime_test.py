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

assignments_to_use=np.where(cv_assignment[:,0]>0)[0]
A=A[assignments_to_use,:]
A=A[:,assignments_to_use]

#ss=np.linspace(0,len(assignments_to_use)-1,len(assignments_to_use))
#np.random.shuffle(ss)
#A=A[ss.astype(int),:]

co_cluster_count, overlap = many_co_cluster_match(A, n_groups)

fig=plt.figure()
plt.subplot(1,2,1); plt.imshow(A);
plt.subplot(1,2,2); plt.imshow(co_cluster_count);
plt.show()