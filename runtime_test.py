import numpy as np
from utils import (
    contingency_matrix,
    get_co_cluster_count,
    get_maxmatches,
    check_equality,
    percent_overlap_vectors,
)
import pickle
import csv
import matplotlib.pyplot as plt

n_groups = 6
discretion_level = 0.2

pkl_filename = "/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_svc_sallcluslabels_fold1.pkl"
with open(pkl_filename, "rb") as file:
    A = pickle.load(file)
A = np.squeeze(A[:, n_groups - 2, :])

final_assignment = []

co_cluster_count = get_co_cluster_count(A)
maxmatches, findmax = get_maxmatches(A, co_cluster_count, 1 - discretion_level)

for n in range(maxmatches.shape[1]):
    if len(final_assignment) > 0:
        tmp_match_pct = np.zeros(len(final_assignment))
        for clus in range(len(final_assignment)):
            tmp_match_pct[clus] = percent_overlap_vectors(
                maxmatches[:, n], final_assignment[clus]
            )
        if np.max(tmp_match_pct) < discretion_level * 100:
            final_assignment.append(maxmatches[:, n])
    else:
        final_assignment.append(maxmatches[:, n])
print(len(final_assignment))


## calculate PAC

# now we have a base of vectors to work from. these might be more than
# the n_groups but represent the distinct different number of groupings found in the data

# identical_values, overlap, non_overlap = check_equality(maxmatches)


# finalmatches = np.full([A.shape[1], n_groups], np.nan)
# if maxmatches.shape[1] == 1:
#    finalmatches[:, 0] = maxmatches

# maxmatches, equal_check = many_co_cluster_match(A, n_groups, 0.5)

output = 0
if output == 1:
    print(A.shape)
    print(co_cluster_count.shape)
    print(maxmatches.shape)
    print(findmax)
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(A)
    plt.colorbar()
    plt.title("input data")
    plt.subplot(1, 3, 2)
    plt.imshow(co_cluster_count)
    plt.colorbar()
    plt.title("number of pair-wise matches")
    plt.subplot(1, 3, 3)
    plt.imshow(maxmatches[:25, :])
    plt.colorbar()
    plt.title("maxmatches")
    plt.show()
elif output == 2:
    fig = plt.figure()
    plt.imshow(identical_values)
    plt.show()


import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from looco_loop import looco_loop
from utils import colorscatter
import scipy

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target

ss=np.linspace(0,len(Y)-1,len(Y))
np.random.shuffle(ss)
X=X[ss.astype(int),:]
Y=Y[ss.astype(int)]

(
    looco_all_clus_labels,
    looco_bic,
    looco_sil,
    looco_cal,
    looco_auc,
    looco_f1,
    looco_betas,
) = looco_loop(X, "full", 1)

fig = plt.figure(figsize=[15, 5])
colorscatter(X, Y, np.ones(shape=X.shape[0]),plt.subplot(1, 2, 1))
all_assigs_mode=[scipy.stats.mode(looco_all_clus_labels[:,p])[0][0].astype(int)+1 for p in range(looco_all_clus_labels.shape[1])]
print(all_assigs_mode)
colorscatter(X, all_assigs_mode, np.ones(shape=X.shape[0]),plt.subplot(1, 2, 2))
plt.show()