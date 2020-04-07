import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from itertools import permutations

from clusmetwrapper import cluster
from looco_loop import loocv_loop
from loocv_assigmatcher_nov import get_co_cluster_count, get_consensus_labels, get_maxmatches, get_final_assignment, \
    match_assignments_to_final_assignments
import matplotlib.pyplot as plt
from utils import colorscatter, max_min_val_check, create_ref_data
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

do_test=2;

if do_test==1:
    ############################################################
    # first basic test - analysis iteration assignment matcher
    print('test: same assignment with labels switched')
    perm = list(permutations([0,1,2]))
    A=np.full([len(perm),len(Y)],np.nan)
    for p in range(len(perm)):
        for g in range(3):
            A[p,np.where(Y==g)[0]]=perm[p][g]

    consensus_label = get_consensus_labels(A,3,1)
    print('rand score for retrieved assignment and original:',sklearn.metrics.adjusted_rand_score(consensus_label,Y))
    print()


    print('test: randomly permuted assignments')
    perm = list(permutations([0,1,2]))
    A=np.full([len(perm),len(Y)],np.nan)
    for p in range(len(perm)):
        for g in range(3):
            A[p,np.where(Y==g)[0]]=perm[p][g]
        np.random.shuffle(A[p,:])
    consensus_label = get_consensus_labels(A,3,1)
    print('rand score for retrieved assignment and original:',sklearn.metrics.adjusted_rand_score(consensus_label,Y))

elif do_test==2:
    ############################################################
    # second basic test - clusters retrieved
    print('test: match for clusters retrieved with LOOCV but no k-fold CV')

    mod = cluster(X,5,np.zeros(shape=[150, 1]),0,1,'full')
    mod.run()


    print(bic)

elif do_test==3:
    ref_data = create_ref_data(X)



