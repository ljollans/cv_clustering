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
mars=sample_descriptors['mars'].to_numpy()
venla=sample_descriptors['venla'].to_numpy()
gsk=sample_descriptors['gsk'].to_numpy()
n_studies=mars+venla+gsk

co_cluster_count, overlap = many_co_cluster_match(A, n_groups)#

just_mars=np.where((mars==1) & (n_studies==1))[0]
just_venla=np.where((venla==1) & (n_studies==1))[0]
just_gsk=np.where((gsk==1) & (n_studies==1))[0]
mars_gsk=np.where((mars==1) & (gsk==1))[0]
mars_venla=np.where((mars==1) & (venla==1))[0]

fig=plt.figure()
plt.subplot(2,3,1); plt.imshow(co_cluster_count); plt.colorbar(); plt.title('all')

plt.subplot(2,3,2); xx=co_cluster_count[just_mars,:];
plt.imshow(xx[:,just_mars]); plt.colorbar(); plt.title('just_mars')

plt.subplot(2,3,3); xx=co_cluster_count[just_gsk,:];
plt.imshow(xx[:,just_gsk]); plt.colorbar(); plt.title('just_gsk')

plt.subplot(2,3,4); xx=co_cluster_count[mars_gsk,:];
plt.imshow(xx[:,mars_gsk]); plt.colorbar(); plt.title('mars_and_gsk')

plt.subplot(2,3,5); xx=co_cluster_count[just_venla,:];
plt.imshow(xx[:,just_venla]); plt.colorbar(); plt.title('just_venla')

plt.subplot(2,3,6); xx=co_cluster_count[mars_venla,:];
plt.imshow(xx[:,mars_venla]); plt.colorbar(); plt.title('mars_and_venla')

plt.show()
#fig=plt.figure()
#plt.subplot(3,3,1); plt.imshow(A); plt.colorbar()
#plt.subplot(3,3,2); plt.imshow(co_cluster_count); plt.colorbar()
#for i in range(3):
#    xx=co_cluster_count[np.where(scanstudy==i+1)[0],:]
#    xx=xx[:,np.where(scanstudy==i+1)[0]]
#    plt.subplot(3,3,4+i); plt.imshow(xx); plt.colorbar()
#for i in range(3):
#    xx=co_cluster_count[np.where(scanstudy==i+1)[0],:]
#    xx=xx[:,np.where(scanstudy!=i+1)[0]]
#    plt.subplot(3,3,7+i); plt.imshow(xx); plt.colorbar()
#plt.show()