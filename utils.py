import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
import copy


def identify_set_and_fold(current_proc, n_cv_folds):
    folds_per_run = n_cv_folds * n_cv_folds
    current_set = np.floor_divide(current_proc, folds_per_run)
    current_set_count_start = current_set * folds_per_run
    current_fold = current_proc - current_set_count_start
    return current_set, current_fold


def select_trainset(cv_assignment,mainfold,subfold):
    return np.where(
    (cv_assignment[:, mainfold] != subfold)
    & (~np.isnan(cv_assignment[:, mainfold]))
)[0]


def select_testset(X,cv_assignment,mainfold,subfold):
    return np.where(
    (cv_assignment[:, mainfold] == subfold)
    & (~np.isnan(cv_assignment[:, mainfold]))
)[0]


def colorscatter(X_e, Y, d, ax):
    try:
        groups = set(Y[~np.isnan(Y)])
    except:
        groups = np.unique(Y)
    colors = matplotlib.cm.tab10(np.linspace(0, 1, 10))
    ctr = -1
    for g in groups:
        ctr += 1
        findme = np.where(Y == g.astype(int))[0]
        cc = np.expand_dims(colors[ctr], axis=0)
        ax.scatter(X_e[findme, 0], X_e[findme, 1], c=cc, s=d[findme] * 5)
    plt.legend(groups)


def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate
    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.
        .. versionadded:: 0.18
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix(
        (np.ones(class_idx.shape[0]), (class_idx, cluster_idx)),
        shape=(n_classes, n_clusters),
        dtype=np.int,
    )
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def contingency_clustering_match(assignment1, assignment2):
    contingency = contingency_matrix(assignment1, assignment2)
    unique_assignment1 = set(assignment1)
    assignment_match = np.full([len(unique_assignment1)], np.nan)
    manipulated_contingency = copy.deepcopy(contingency)

    while np.sum(manipulated_contingency) > 0:
        remaining_highest_match = np.max(manipulated_contingency)
        loc_highest_match = np.where(manipulated_contingency == remaining_highest_match)

        if len(loc_highest_match[0]) == 1:
            assignment_match[loc_highest_match[0]] = loc_highest_match[1]
            manipulated_contingency[loc_highest_match[0], :] = 0
            manipulated_contingency[:, loc_highest_match[1]] = 0

        else:
            unique_loc0 = np.unique(loc_highest_match[0])
            unique_loc1 = np.unique(loc_highest_match[1])
            if (
                len(unique_loc0)
                == len(loc_highest_match[0]) & len(unique_loc1)
                == len(loc_highest_match[1])
            ):
                assignment_match[loc_highest_match[0][0]] = loc_highest_match[1][0]
                manipulated_contingency[loc_highest_match[0][0], :] = 0
                manipulated_contingency[:, loc_highest_match[1][0]] = 0
            # # loop over instances of the max value to check if they conflict
            # for i in range(len(loc_highest_match[0])):
            #     instances_row_match = len(np.where(loc_highest_match[0] == loc_highest_match[0][i]))
            #     instances_col_match = len(np.where(loc_highest_match[1] == loc_highest_match[1][i]))
            #     if instances_col_match == 1 & instances_row_match == 1:
            #         assignment_match[loc_highest_match[0][i], loc_highest_match[1][i]] = counter
            #         manipulated_contingency[loc_highest_match[0][i], loc_highest_match[1][i]] = 0
            #         counter += 1
            # if np.max(manipulated_contingency) == remaining_highest_match:
            else:
                break

    return contingency, assignment_match, manipulated_contingency


def get_co_cluster_count(A):
    co_cluster_count = np.full([A.shape[1], A.shape[1]], 0)
    for a1 in range(A.shape[1]):
        for a2 in range(A.shape[1]):
            if a1 > a2:
                co_cluster_count[a1, a2] = len(np.where(A[:, a1] == A[:, a2])[0])
    np.fill_diagonal(co_cluster_count, 0)
    return co_cluster_count

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_pac(con_mat):
    x1 = 0.1
    x2 = 0.9

    p = con_mat[np.tril_indices(con_mat.shape[0])].flatten()

    xs, ys = ecdf(p)
    select_vals = np.where(np.logical_and(xs >= x1, xs <= x2))
    x1_x2_range = ys[select_vals[0]]

    x1_val = x1_x2_range[0]
    x2_val = x1_x2_range[-1]
    pac = x2_val - x1_val

    return pac

def get_maxmatches(A, co_cluster_count, ratio_of_max_as_lower):
    max_match = np.max(co_cluster_count) * ratio_of_max_as_lower
    findmax = np.where(co_cluster_count >= max_match)
    print(
        "highest match frequency=",
        str(max_match),
        "occurring for",
        str(len(findmax[0])),
        "pairs.",
    )
    maxmatches = np.full([A.shape[1], len(findmax[0]),], np.nan)
    for n in range(len(findmax[0])):
        a1 = findmax[0][n]
        a2 = findmax[1][n]
        matches = np.where(A[:, a1] == A[:, a2])[0]
        maxmatches[matches, n] = A[matches, a1]
    return maxmatches, findmax


def percent_overlap_vectors(a1, a2):

    return ((len(np.where(a2 - a1 == 0)[0])) * 100) / (
        a1.shape[0] - len(np.where(np.isnan(a1))[0])
    )




def check_equality(maxmatches):
    identical_values = np.full([maxmatches.shape[1], maxmatches.shape[1]], np.nan)
    for n in range(maxmatches.shape[1]):
        for m in range(maxmatches.shape[1]):
            identical_values[n, m] = len(
                np.where(maxmatches[:, m] - maxmatches[:, n] == 0)[0]
            )
            identical_values[n, m] = (identical_values[n, m] * 100) / (
                maxmatches.shape[0] - len(np.where(np.isnan(maxmatches[:, n]))[0])
            )
    overlap = np.where(identical_values > 75)
    non_overlap = np.where(identical_values < 25)
    return identical_values, overlap, non_overlap
