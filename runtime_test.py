import numpy as np
from utils import (
    many_co_cluster_match,
    contingency_matrix,
    get_co_cluster_count,
    get_maxmatches,
    check_equality)
import pickle
import csv
import matplotlib.pyplot as plt

cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

n_groups = 6

pkl_filename = "/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_svc_sallcluslabels_fold10.pkl"
with open(pkl_filename, "rb") as file:
    A = pickle.load(file)
A = np.squeeze(A[:, n_groups - 2, :])

co_cluster_count = get_co_cluster_count(A)
maxmatches, findmax = get_maxmatches(A, co_cluster_count, 0.95)

# all_findmax = np.append(findmax[0], findmax[1])
# uq_findmax = np.unique(all_findmax)

# ct_uq_findmax = np.array([len(np.where(all_findmax == n)[0]) for n in uq_findmax])
# max_ct_uq_findmax = uq_findmax[np.where(ct_uq_findmax == np.max(ct_uq_findmax))[0]]

identical_values, overlap, non_overlap = check_equality(maxmatches)


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
elif output==2:
    fig = plt.figure ()
    plt.imshow ( identical_values )
    plt.show()