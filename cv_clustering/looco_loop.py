import numpy as np
from cv_clustering.clusmets import calculate_clustering_metrics


def loocv_loop(x_all, covariance, k):

    n_samples = x_all.shape[0]
    n_features = x_all.shape[1]

    loocv_all_clus_labels = np.full([n_samples, n_samples], np.nan)
    loocv_bic = np.full([n_samples], np.nan)
    loocv_sil = np.full([n_samples], np.nan)
    loocv_cal = np.full([n_samples], np.nan)
    loocv_auc = np.full([n_samples], np.nan)
    loocv_f1 = np.full([n_samples], np.nan)
    loocv_betas = np.full([n_samples, n_features, k + 2], np.nan)
    loocv_n_across_cv_folds=np.full([n_samples,k+2],np.nan)

    for loocv in range(n_samples):
        print('loocv i=',loocv)
        x = np.delete(x_all, loocv, axis=0)
        save_vector = np.delete(np.arange(n_samples),loocv)
        [
                loocv_all_clus_labels[loocv, save_vector],
                loocv_bic[loocv],
                loocv_sil[loocv],
                loocv_cal[loocv],
                loocv_auc[loocv],
                loocv_f1[loocv],
                loocv_betas[loocv, :, :],
                loocv_n_across_cv_folds[loocv,:]
            ] = calculate_clustering_metrics({'data':x, 'nclus':k, 'covariance':covariance})

    return (
        loocv_all_clus_labels,
        loocv_bic,
        loocv_sil,
        loocv_cal,
        loocv_auc,
        loocv_f1,
        loocv_betas,
        loocv_n_across_cv_folds
    )
