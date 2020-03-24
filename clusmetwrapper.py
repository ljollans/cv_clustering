#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os

from looco_loop import looco_loop

def clusmetwrapper(design):

    X = design["X"]

    n_samples = X.shape[0]
    n_features = X.shape[1]

    all_clus_labels = np.full([n_samples, design["n_ks"], n_samples], np.nan)
    bic = np.full([n_samples, design["n_ks"]], np.nan)
    sil = np.full([n_samples, design["n_ks"]], np.nan)
    cal = np.full([n_samples, design["n_ks"]], np.nan)
    auc = np.full([n_samples, design["n_ks"]], np.nan)
    f1 = np.full([n_samples, design["n_ks"]], np.nan)
    betas = np.full([n_samples, design["n_ks"], design["n_ks"] + 2, n_features], np.nan)

    cv_assignment = design["cv_assignment"]
    train_index_sub = np.where(
        (cv_assignment[:, design["mainfold"]] != design["subfold"])
        & (~np.isnan(cv_assignment[:, design["mainfold"]]))
    )[0]
    x2 = X[train_index_sub, :]

    for nclus in range(design["n_ks"]):
        (
            tmp_all_clus_labels,
            bic[train_index_sub, nclus],
            sil[train_index_sub, nclus],
            cal[train_index_sub, nclus],
            auc[train_index_sub, nclus],
            f1[train_index_sub, nclus],
            betas[train_index_sub, nclus, :nclus+2, :]
        ) = looco_loop(x2, design["covariance"], nclus)
        for t in range(len(train_index_sub)):
            all_clus_labels[train_index_sub[t],nclus,train_index_sub]=tmp_all_clus_labels[t,:]

    return all_clus_labels, bic, sil, cal, auc, f1, betas
