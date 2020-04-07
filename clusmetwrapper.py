#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os

from looco_loop import loocv_loop


class cluster:

    def __init__(self, X, n_ks, cv_assignment, mainfold, subfold, covariance):

        self.data = X
        self.nk = n_ks
        self.cv_assignment = cv_assignment
        self.mainfold = mainfold
        self.subfold = subfold
        self.covariance = covariance

    def run(self):

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        self.all_clus_labels = np.full([self.n_samples, self.nk, self.n_samples], np.nan)
        self.bic = np.full([self.n_samples, self.nk], np.nan)
        self.sil = np.full([self.n_samples, self.nk], np.nan)
        self.cal = np.full([self.n_samples, self.nk], np.nan)
        self.auc = np.full([self.n_samples, self.nk], np.nan)
        self.f1 = np.full([self.n_samples, self.nk], np.nan)
        self.betas = np.full([self.n_samples, self.nk,  self.n_features, self.nk + 2], np.nan)
        self.n_per_classification = np.full([self.n_samples, self.nk, self.nk + 2], np.nan)

        self.train_index_sub = np.where(
            (self.cv_assignment[:, self.mainfold] != self.subfold)
            & (~np.isnan(self.cv_assignment[:, self.mainfold]))
        )[0]
        x2 = self.data[self.train_index_sub, :]

        for nclus in range(self.nk):
            print(nclus+2,'clusters')
            (
                tmp_all_clus_labels,
                self.bic[self.train_index_sub, nclus],
                self.sil[self.train_index_sub, nclus],
                self.cal[self.train_index_sub, nclus],
                self.auc[self.train_index_sub, nclus],
                self.f1[self.train_index_sub, nclus],
                self.betas[self.train_index_sub, nclus, :, :nclus + 2],
                self.n_per_classification[self.train_index_sub,nclus,:nclus+2]
            ) = loocv_loop(x2, self.covariance, nclus)
            for t in range(len(self.train_index_sub)):
                self.all_clus_labels[self.train_index_sub[t],nclus,self.train_index_sub]=tmp_all_clus_labels[t,:]

