import numpy as np
import matplotlib.pyplot as plt
from utils import many_co_cluster_match, contingency_matrix
import time
import pickle
import csv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
from scipy import stats

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
just_mars=np.where((mars==1) & (n_studies==1))[0]
just_venla=np.where((venla==1) & (n_studies==1))[0]
just_gsk=np.where((gsk==1) & (n_studies==1))[0]
mars_gsk=np.where((mars==1) & (gsk==1))[0]
mars_venla=np.where((mars==1) & (venla==1))[0]


co_cluster_count, overlap = many_co_cluster_match(A, n_groups)#
pca=PCA(n_components=6)
pca.fit(co_cluster_count)
X = pca.transform(co_cluster_count)

fig=plt.figure()
plt.subplot(3,3,1); plt.imshow(co_cluster_count); plt.colorbar()
plt.subplot(3,3,2); plt.plot(pca.explained_variance_);
for component in range(6):
    plt.subplot(3,3,3+component); plt.plot(X[:,component]);
plt.show()

titles=['scanstudy','just_mars','just_gsk','just_venla','mars_gsk','mars_venla']
x_variables=[scanstudy,just_mars,just_gsk,just_venla,mars_gsk,mars_venla]
fig=plt.figure()
ctr=0
for component in range(4):
    for x_variable_loop in range(6):
        ctr+=1
        if x_variable_loop>0:
            y=np.zeros(shape=[398]); y[x_variables[x_variable_loop]]=1
            rvs1 = X[np.where(y==0)[0],component]
            rvs2 = X[np.where(y == 1)[0], component]
            str2use = (titles[x_variable_loop],"{:10.4f}".format(stats.ttest_ind(rvs1, rvs2)[1]))
        else:
            y=x_variables[x_variable_loop]
            str2use=titles[x_variable_loop]
        plt.subplot(4,6,ctr);
        sns.violinplot(y,X[:,component]);
        plt.title(str2use)
plt.show()




#fig=plt.figure()
#plt.subplot(2,3,1); plt.imshow(co_cluster_count); plt.colorbar(); plt.title('all')##

#plt.subplot(2,3,2); xx=co_cluster_count[just_mars,:];
#plt.imshow(xx[:,just_mars]); plt.colorbar(); plt.title('just_mars')

#plt.subplot(2,3,3); xx=co_cluster_count[just_gsk,:];
#plt.imshow(xx[:,just_gsk]); plt.colorbar(); plt.title('just_gsk')

#plt.subplot(2,3,4); xx=co_cluster_count[mars_gsk,:];
#plt.imshow(xx[:,mars_gsk]); plt.colorbar(); plt.title('mars_and_gsk')

#plt.subplot(2,3,5); xx=co_cluster_count[just_venla,:];
#plt.imshow(xx[:,just_venla]); plt.colorbar(); plt.title('just_venla')

#plt.subplot(2,3,6); xx=co_cluster_count[mars_venla,:];
#plt.imshow(xx[:,mars_venla]); plt.colorbar(); plt.title('mars_and_venla')

#plt.show()



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