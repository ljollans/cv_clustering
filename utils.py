import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp
import copy

from loocv_assigmatcher_nov import get_maxmatches, get_co_cluster_count


def identify_set_and_fold(current_proc, n_cv_folds):
    folds_per_run = n_cv_folds * n_cv_folds
    current_set = np.floor_divide(current_proc, folds_per_run)
    current_set_count_start = current_set * folds_per_run
    current_fold = current_proc - current_set_count_start
    return current_set, current_fold


def select_trainset(cv_assignment, mainfold, subfold):
    return np.where(
        (cv_assignment[:, mainfold] != subfold)
        & (~np.isnan(cv_assignment[:, mainfold]))
    )[0]


def select_testset(cv_assignment, mainfold, subfold):
    return np.where(
        (cv_assignment[:, mainfold] == subfold)
        & (~np.isnan(cv_assignment[:, mainfold]))
    )[0]


def colorscatter(x, y, d, ax):
    try:
        groups = set(y[~np.isnan(y)])
    except:
        groups = np.unique(y)
    colors = matplotlib.cm.tab10(np.linspace(0, 1, 10))
    ctr = -1
    for g in groups:
        ctr += 1
        findme = np.where(y == g.astype(int))[0]
        cc = np.expand_dims(colors[ctr], axis=0)
        ax.scatter(x[findme, 0], x[findme, 1], c=cc, s=d[findme] * 5)
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



def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def get_pac(con_mat):
    x1 = 0.1
    x2 = 0.9

    p = con_mat.flatten()

    xs, ys = ecdf(p)
    select_vals = np.where(np.logical_and(xs >= x1, xs <= x2))
    x1_x2_range = ys[select_vals[0]]

    x1_val = x1_x2_range[0]
    x2_val = x1_x2_range[-1]
    pac = x2_val - x1_val

    return pac


def n_clus_retrieval_chk(cluster_assignments):
    n_maxcluster = np.zeros(shape=[cluster_assignments.shape[1]])
    pacs = np.zeros(shape=[cluster_assignments.shape[1]])
    for ngroups in range(cluster_assignments.shape[1]):
        cluster_assignments_k = cluster_assignments[:, ngroups, :]
        co_cluster_count = get_co_cluster_count(cluster_assignments_k)
        try:
            pacs[ngroups] = get_pac(co_cluster_count)
        except:
            pacs[ngroups] = 0.0

        maxmatches, findmax = get_maxmatches(cluster_assignments_k, co_cluster_count, 1 - 0.2)

        final_assignment = get_final_assignment(maxmatches, 25)

        n_maxcluster[ngroups] = (len(final_assignment))
    u_cr = np.unique(n_maxcluster)
    cr_count = [len(np.where(n_maxcluster == i)[0]) for i in u_cr]
    max_cr = np.max(cr_count)
    find_max_cr = np.where(cr_count == max_cr)

    return n_maxcluster, pacs, u_cr[find_max_cr[0][0]]



def match_assignments_to_final_assignments(cluster_assignments, final_assignment):
    for n in range(cluster_assignments.shape[0]):
        tmp_match_pct = np.zeros(len(final_assignment))
        for clus in range(len(final_assignment)):
            tmp_match_pct[clus] = percent_overlap_vectors(
                cluster_assignments[], final_assignment[clus]
            )


def percent_overlap_vectors(a1, a2):
    if len(a1)==len(a2):
        n_overlap = len(a1) - len(np.unique(np.append(np.where(np.isnan(a1))[0], np.where(np.isnan(a2))[0])))
        n_matches=len(np.where(a1-a2==0)[0])
        pct_overlap = (n_matches * 100)/n_overlap
    else:
        raise Exception('a1 and a2 must have the same dimensions')
    return pct_overlap





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
