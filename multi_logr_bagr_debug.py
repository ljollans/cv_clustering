import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations

from clusmetwrapper import clusmetwrapper
from looco_loop import loocv_loop
from loocv_assigmatcher_nov import get_co_cluster_count, get_consensus_labels, get_maxmatches, get_final_assignment, \
    match_assignments_to_final_assignments
import matplotlib.pyplot as plt
from utils import colorscatter, max_min_val_check
import scipy
import nimfa

iris = datasets.load_iris()
X = iris.data
Y = iris.target
ss=np.linspace(0,len(Y)-1,len(Y))
np.random.shuffle(ss)
X=X[ss.astype(int),:]
Y=Y[ss.astype(int)]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

(
    tmp_all_clus_labels,
    bic,
    sil,
    cal,
    auc,
    f1,
    betas,
    loocv_n_across_cv_folds
) = loocv_loop(X, 'full', 4)

print(loocv_n_across_cv_folds)