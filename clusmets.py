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


def clusmets(design):
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
    x_ytrain = np.append(np.expand_dims(y, 1), x, axis=1)
    [
        tmpauc,
        tmpf1,
        betas2use,
        overallpred0,
        overallpred,
        aucs_partial,
        f1s_partial,
        tmpbetas,
        groupclass,
        correctclass,
        problemrec,
    ] = multi_logr_bagr(10, x_ytrain, design["nclus"] + 2, 4, 0)
    auc = np.mean(tmpauc)
    f1 = np.mean(tmpf1)
    betas = np.nanmean(tmpbetas, axis=2).T

    return clus_labels, bic, sil, cal, auc, f1, betas
