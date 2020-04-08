import csv
import numpy as np
import time
from clusmetwrapper import cluster
import pickle
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from loocv_assigmatcher_nov import collect_betas_for_corresponding_clus
from utils import rand_score_withnans, ecdf

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

# are assignments based on betas in line with consensus_labels
k=4

data = mod.data[mod.train_index_sub,:]
betas = mod.betas[mod.train_index_sub,k,:k+2,:]
assignments = mod.iteration_assignments[k]

aggregated_betas, new_betas_array = collect_betas_for_corresponding_clus(assignments, betas)

Y = data.dot(aggregated_betas)

fig=plt.figure()
for nclus in range(k+2):
    plt.subplot(3,2,nclus+1)
    sns.distplot(Y[:,nclus])
plt.show()

fig=plt.figure()
for nclus in range(k+2):
    xs = (np.ones(Y.shape[0]) * nclus) + np.random.rand(Y.shape[0])
    plt.scatter(xs,Y[:,nclus])
plt.show()

all_ys = np.full([Y.shape[0], Y.shape[1]],np.nan)
for nclus in range(k+2):
    xs,ys, idx=ecdf(Y[:,nclus])
    all_ys[idx,nclus]=ys


Yt=Y[:25,:]; all_yst=all_ys[:25,:]
fig=plt.figure()
plt.subplot(2,1,1)
for nclus in range(k+2):
    plt.scatter(np.arange(Yt.shape[0]),Yt[:,nclus])
plt.subplot(2,1,2)
for nclus in range(k+2):
    plt.scatter(np.arange(Yt.shape[0]),all_yst[:,nclus])
plt.show()

plt.plot(np.max(all_ys,axis=1))
