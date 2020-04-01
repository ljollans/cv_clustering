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


def percent_overlap_vectors(a1, a2):
    if len(a1)==len(a2):
        n_overlap = len(a1) - len(np.unique(np.append(np.where(np.isnan(a1))[0], np.where(np.isnan(a2))[0])))
        n_matches=len(np.where(a1-a2==0)[0])
        pct_overlap = (n_matches * 100)/n_overlap
    else:
        raise Exception('a1 and a2 must have the same dimensions')
    return pct_overlap


def sdremoved_highest_val(fit_vals):
    a=fit_vals.shape
    if a[1]>a[0]:
        fit_vals=fit_vals.T
    clear_max1 = np.where((np.max(fit_vals,axis=1)-(np.mean(fit_vals,axis=1)+1*np.std(fit_vals,axis=1)))>0)[0]
    clear_max2 = np.where((np.max(fit_vals, axis=1) - (np.mean(fit_vals, axis=1) + 2 * np.std(fit_vals, axis=1))) > 0)[
        0]
    clear_max3 = np.where((np.max(fit_vals, axis=1) - (np.mean(fit_vals, axis=1) + 3 * np.std(fit_vals, axis=1))) > 0)[
        0]
    return clear_max1, clear_max2, clear_max3

def max_min_val_check(fit_vals, max_thresh,min_thresh):
    a = fit_vals.shape
    if a[1] > a[0]:
        fit_vals = fit_vals.T
    m1 = np.sum(((fit_vals - max_thresh) > 0), axis=1)
    m2 = np.sum(((fit_vals - min_thresh) < 0), axis=1)
    f = np.where((m1 == 1) & (m2 ==fit_vals.shape[1] - 1))
    return f[0]




