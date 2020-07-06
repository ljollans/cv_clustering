import matplotlib
import matplotlib.pyplot as plt
from scipy import sparse as sp
import scipy
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score, silhouette_score
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


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

def colorscatter(x, y, d, ax, legendyn=1, aa=1, mm='o'):
    try:
        groups = set(y[~np.isnan(y)])
    except:
        groups = np.unique(y)
    if len(groups)>10:
        colors = matplotlib.cm.tab20(np.linspace(0, 1, 20))
    else:
        colors = matplotlib.cm.tab10(np.linspace(0, 1, 10))
    ctr = -1
    for g in groups:
        ctr += 1
        findme = np.where(y == g.astype(int))[0]
        cc = np.expand_dims(colors[g.astype(int)], axis=0)
        ax.scatter(x[findme, 0], x[findme, 1], c=cc, s=d[findme] * 5, alpha=aa, marker=mm)
    if legendyn==1:
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
    idx = np.argsort(x)
    xs = x[idx]
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys, idx


def get_pac(con_mat):
    x1 = 0.1
    x2 = 0.9

    p = con_mat.flatten()

    xs, ys, idx = ecdf(p)
    select_vals = np.where(np.logical_and(xs >= x1, xs <= x2))
    x1_x2_range = ys[select_vals[0]]

    if len(x1_x2_range)==0:
        pac = 0
        print('pac set to 0. Total n =',str(len(ys)), '. n <', str(x1),'=', str(len(np.where(xs<x1)[0])),';  n >',str(x2),'=', str(len(np.where(xs>x2)[0])))
    else:
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

def rand_score_withnans(a1,a2):
    a1 = np.delete(a1, np.where(np.isnan(a2))[0])
    a2 = np.delete(a2, np.where(np.isnan(a2))[0])
    a2 = np.delete(a2, np.where(np.isnan(a1))[0])
    a1 = np.delete(a1, np.where(np.isnan(a1))[0])
    return adjusted_rand_score(a1,a2)


def ttest_withnans(a1,a2):
    a1 = np.delete(a1, np.where(np.isnan(a1))[0])
    a2 = np.delete(a2, np.where(np.isnan(a2))[0])
    return scipy.stats.ttest_ind(a1,a2)

def silhouette_score_withnans(X,a):
    X = np.delete(X, np.where(np.isnan(a))[0], axis=0)
    a = np.delete(a, np.where(np.isnan(a))[0])
    return silhouette_score(X,a)

def mutual_info_score_withnans(a1,a2):
    a1 = np.delete(a1, np.where(np.isnan(a2))[0])
    a2 = np.delete(a2, np.where(np.isnan(a2))[0])
    a2 = np.delete(a2, np.where(np.isnan(a1))[0])
    a1 = np.delete(a1, np.where(np.isnan(a1))[0])
    return adjusted_mutual_info_score(a1,a2)


def vmeasure_withnans(a1,a2):
    a1 = np.delete(a1, np.where(np.isnan(a2))[0])
    a2 = np.delete(a2, np.where(np.isnan(a2))[0])
    a2 = np.delete(a2, np.where(np.isnan(a1))[0])
    a1 = np.delete(a1, np.where(np.isnan(a1))[0])
    return v_measure_score(a1,a2)


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


def create_ref_data(x, random_state=None):
    '''
    A function that returns a reference data set without any cluster structure.
    This function was translated from the "M3C" R package from Christopher John and David Watson (John et al., SciRep, 2020).
    The approach bases on Tibshirani et al. (2001) and preserves covariance structure of the data via PCA.

    Args:
        x: your data, can be a pandas dataframe or a matrix. rows should be the samples, columns the features.

    Returns:
        ref_data: reference data set for your data, same type (dataframe or matrix)
    '''

    # if dataframe is input, convert to matrix
    in_df = 0
    if isinstance(x, pd.core.frame.DataFrame):
        df_names = x.columns
        in_df = 1
        x = x.values

    # rows and columns
    r = x.shape[0]
    c = x.shape[1]

    # perform PCA
    pca = PCA()
    pca.fit(x)

    # rotated data
    x_t = pca.fit_transform(x.T)

    # function to get PCA covariance matrix and SDs
    def colSdColMeans(x):
        n = x.shape[0]
        colVar = np.mean(x * x, axis=0) - (np.mean(x, axis=0) ** 2)
        return np.sqrt(colVar * n / (n - 1))

    # get SDs from colSdColMeans function
    sds = colSdColMeans(x_t)

    # simulated data based on normal distributions: mean 0, sd from sds
    if random_state:
        np.random.seed(random_state)
    sim_data = np.random.normal(0, sds, (c, c))

    # create reference dataset
    ref_data = (np.matmul(sim_data, pca.components_) + pca.mean_).T

    # possibly back to dataframe
    if in_df:
        ref_data = pd.DataFrame(ref_data, columns=df_names)

    return ref_data

def coph_cor(consensus_matrix):
    """
    Compute cophenetic correlation coefficient of consensus matrix, generally obtained from multiple NMF runs.

    The cophenetic correlation coefficient is measure which indicates the dispersion of the consensus matrix and is based
    on the average of connectivity matrices. It measures the stability of the clusters obtained from NMF.
    It is computed as the Pearson correlation of two distance matrices: the first is the distance between samples induced by the
    consensus matrix; the second is the distance between samples induced by the linkage used in the reordering of the consensus
    matrix [Brunet2004]_.

    Return real number. In a perfect consensus matrix, cophenetic correlation equals 1. When the entries in consensus matrix are
    scattered between 0 and 1, the cophenetic correlation is < 1. We observe how this coefficient changes as factorization rank
    increases. We select the first rank, where the magnitude of the cophenetic correlation coefficient begins to fall [Brunet2004]_.

    :param idx: Used in the multiple NMF model. In factorizations following standard NMF model or nonsmooth NMF model
                :param:`idx` is always None.
    :type idx: None or `str` with values 'coef' or 'coef1' (`int` value of 0 or 1, respectively)
    """

    # upper diagonal elements of consensus
    avec = np.array([consensus_matrix[i, j] for i in range(consensus_matrix.shape[0] - 1)
                     for j in range(i + 1, consensus_matrix.shape[1])])
    # consensus entries are similarities, conversion to distances
    Y = 1 - avec
    Z = scipy.cluster.hierarchy.linkage(Y, method='average')
    # cophenetic correlation coefficient of a hierarchical clustering
    # defined by the linkage matrix Z and matrix Y from which Z was
    # generated
    return scipy.cluster.hierarchy.cophenet(Z, Y)[0]

def gmm_js(gmm_p, gmm_q, n_samples=10**5):
    X = gmm_p.sample(n_samples)
    log_p_X, _ = gmm_p.score_samples(X)
    log_q_X, _ = gmm_q.score_samples(X)
    log_mix_X = np.logaddexp(log_p_X, log_q_X)

    Y = gmm_q.sample(n_samples)
    log_p_Y, _ = gmm_p.score_samples(Y)
    log_q_Y, _ = gmm_q.score_samples(Y)
    log_mix_Y = np.logaddexp(log_p_Y, log_q_Y)

    return (log_p_X.mean() - (log_mix_X.mean() - np.log(2))
            + log_q_Y.mean() - (log_mix_Y.mean() - np.log(2))) / 2


def get_gradient_change(v):
    v_change = [v[i+1]-v[i] for i in range(len(v)-1)]
    return v_change


def linear_multiclass_calibrated(Xtrain,ytrain, Xtest):
    clf = LogisticRegression(random_state=0, multi_class='ovr').fit(Xtrain, ytrain)
    clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic').fit(Xtrain, ytrain)
    return clf_isotonic.predict_proba(Xtest), clf.coef_, clf.intercept_


def do_all_clus_corrs(all_subfold_betas, itcpt):
    ncv = len(all_subfold_betas)
    nclus = all_subfold_betas[0].shape[1]
    allcorrs = np.full([ncv, ncv, nclus, nclus], np.nan)
    for sf1 in range(ncv):
        for sf2 in range(ncv):
            if sf1 != sf2:
                for clus1 in range(nclus):
                    for clus2 in range(nclus):
                        if itcpt==1:
                            a=all_subfold_betas[sf1][1:, clus1]
                            b=all_subfold_betas[sf2][1:, clus2]
                        else:
                            a = all_subfold_betas[sf1][:, clus1]
                            b = all_subfold_betas[sf2][:, clus2]
                        allcorrs[sf1, sf2, clus1, clus2] = np.corrcoef(a,b)[0,1]
    return allcorrs