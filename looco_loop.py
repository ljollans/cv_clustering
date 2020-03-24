import numpy as np
import clusmets
from clusmets import clusmets


def looco_loop(x_all, covariance, k):

    n_samples = x_all.shape[0]
    n_features = x_all.shape[1]

    looco_all_clus_labels = np.full([n_samples, n_samples], np.nan)
    looco_bic = np.full([n_samples], np.nan)
    looco_sil = np.full([n_samples], np.nan)
    looco_cal = np.full([n_samples], np.nan)
    looco_auc = np.full([n_samples], np.nan)
    looco_f1 = np.full([n_samples], np.nan)
    looco_betas = np.full([n_samples, k + 2, n_features], np.nan)

    for looco in range(n_samples):
        x = np.delete(x_all, looco, axis=0)
        save_vector = np.delete(np.arange(n_samples),looco)
        [
            looco_all_clus_labels[looco, save_vector],
            looco_bic[looco],
            looco_sil[looco],
            looco_cal[looco],
            looco_auc[looco],
            looco_f1[looco],
            looco_betas[looco, :, :],
        ] = clusmets([x, k, covariance])

    return (
        looco_all_clus_labels,
        looco_bic,
        looco_sil,
        looco_cal,
        looco_auc,
        looco_f1,
        looco_betas,
    )
