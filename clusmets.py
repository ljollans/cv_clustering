#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:14:07 2020

@author: lee_jollans
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from multi_logr_bag import multi_logr_bagr


def calculate_clustering_metrics(design):
    gmm = GaussianMixture(
        n_components=design["nclus"] + 2,
        covariance_type=design["covariance"],
        n_init=100,
        random_state=10,
    ).fit(design["data"])
    clus_labels = gmm.predict(design["data"])
    bic = gmm.bic(design["data"])
    sil = metrics.silhouette_score(design["data"], clus_labels)
    cal = metrics.calinski_harabasz_score(design["data"], clus_labels)

    x = design["data"][np.where(clus_labels != -1)[0], :]
    y = clus_labels[np.where(clus_labels != -1)[0]]
    x_y_train = np.append(np.expand_dims(y, 1), x, axis=1)
    try:
        (
            auc_across_cv_folds,
            f1_across_cv_folds,
            beta_avg_across_folds,
            overall_prediction_continuous,
            overall_prediction_discrete,
            auc_per_cv_fold,
            f1_per_cv_fold,
            betas_per_fold,
        ) = multi_logr_bagr(10, x_y_train, design["nclus"] + 2, 4, 0)
    except:
        multi_logr_bagr_debug(10, x_y_train, design["nclus"] + 2, 4, 0)

    return clus_labels, bic, sil, cal, np.mean(auc_across_cv_folds), np.mean(f1_across_cv_folds), beta_avg_across_folds
