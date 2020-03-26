import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse as sp


def identify_set_and_fold(current_proc, n_cv_folds):
    folds_per_run = n_cv_folds * n_cv_folds
    current_set = np.floor_divide(current_proc, folds_per_run)
    current_set_count_start = current_set * folds_per_run
    current_fold = current_proc - current_set_count_start
    return current_set, current_fold

def colorscatter(X_e,Y,d,ax):
    try:
        groups=(set(Y[~np.isnan(Y)]))
    except:
        groups=np.unique(Y)
    colors = matplotlib.cm.tab10(np.linspace(0, 1, 10))
    ctr=-1
    for g in groups:
        ctr+=1
        findme=np.where(Y==g.astype(int))[0]
        cc=np.expand_dims(colors[ctr],axis=0)
        ax.scatter(X_e[findme,0],X_e[findme,1],c=cc, s=d[findme]*5)
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
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def co_clustering_match(assignment1, assignment2):
    contingency=contingency_matrix(assignment1, assignment2)
    unique_assignment1=set(assignment1)
    unique_assignment2=set(assignment2)

    assignment_match=np.full([len(unique_assignment1),len(unique_assignment2)],np.nan)

    counter=0
    while np.sum(contingency)>0:
        remaining_highest_match=np.max(contingency)
        loc_highest_match=np.where(contingency==remaining_highest_match)
        assignment_match[loc_highest_match[0],loc_highest_match[1]]=counter
        counter+=1

    return contingency, assignment_match



