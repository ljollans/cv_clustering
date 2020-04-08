#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn

from looco_loop import loocv_loop
from loocv_assigmatcher_nov import (
    n_clus_retrieval_chk,
    n_clus_retrieval_grid,
    get_co_cluster_count,
    get_consensus_labels,
    infer_iteration_clusmatch)
from utils import coph_cor, rand_score_withnans
import sys

sys.path.append(
    "/Users/lee_jollans/PycharmProjects/Cluster_Ensembles/src/Cluster_Ensembles"
)
import Cluster_Ensembles as CE


class cluster:
    def __init__(self, X, n_ks, cv_assignment, mainfold, subfold, covariance):

        self.data = X
        self.nk = n_ks
        self.cv_assignment = cv_assignment
        self.mainfold = mainfold
        self.subfold = subfold
        self.covariance = covariance

        self.n_samples = self.data.shape[0]
        self.n_features = self.data.shape[1]

        self.all_clus_labels = np.full(
            [self.n_samples, self.nk, self.n_samples], np.nan
        )
        self.bic = np.full([self.n_samples, self.nk], np.nan)
        self.sil = np.full([self.n_samples, self.nk], np.nan)
        self.cal = np.full([self.n_samples, self.nk], np.nan)
        self.auc = np.full([self.n_samples, self.nk], np.nan)
        self.f1 = np.full([self.n_samples, self.nk], np.nan)
        self.betas = np.full(
            [self.n_samples, self.nk, self.n_features, self.nk + 2], np.nan
        )
        self.n_per_classification = np.full(
            [self.n_samples, self.nk, self.nk + 2], np.nan
        )

        self.train_index_sub = np.where(
            (self.cv_assignment[:, self.mainfold] != self.subfold)
            & (~np.isnan(self.cv_assignment[:, self.mainfold]))
        )[0]

    def run(self):

        x2 = self.data[self.train_index_sub, :]

        for nclus in range(self.nk):
            print(nclus + 2, "clusters")
            (
                tmp_all_clus_labels,
                self.bic[self.train_index_sub, nclus],
                self.sil[self.train_index_sub, nclus],
                self.cal[self.train_index_sub, nclus],
                self.auc[self.train_index_sub, nclus],
                self.f1[self.train_index_sub, nclus],
                self.betas[self.train_index_sub, nclus, :, : nclus + 2],
                self.n_per_classification[self.train_index_sub, nclus, : nclus + 2],
            ) = loocv_loop(x2, self.covariance, nclus)
            for t in range(len(self.train_index_sub)):
                self.all_clus_labels[
                    self.train_index_sub[t], nclus, self.train_index_sub
                ] = tmp_all_clus_labels[t, :]

    def pull_in_saves(self, bic, sil, cal, all_clus_labels, auc, f1, betas):
        self.bic = bic
        self.sil = sil
        self.cal = cal
        self.all_clus_labels = all_clus_labels
        self.auc = auc
        self.f1 = f1
        self.betas = betas

    def plot_loocv(self):
        df = pd.DataFrame({}, columns=["bic", "sil", "auc", "f1", "k", "ppt"])
        for nclus in range(self.nk):
            tmp_df = pd.DataFrame(
                {
                    "bic": np.squeeze(self.bic[:, nclus]),
                    "sil": np.squeeze(self.sil[:, nclus]),
                    "auc": np.squeeze(self.auc[:, nclus]),
                    "f1": np.squeeze(self.f1[:, nclus]),
                    "k": np.ones(shape=[398]) * nclus + 2,
                    "ppt": np.arange(398),
                },
                columns=["bic", "sil", "auc", "f1", "k", "ppt"],
            )
            df = df.append(tmp_df)

        sns.set()
        fig = plt.figure()
        plot_these = ["bic", "sil", "auc", "f1"]
        call_these = ["BIC", "Silhouette score", "AUC", "F1 score"]
        for p in range(len(plot_these)):
            plt.subplot(2, 2, p + 1)
            sns.violinplot("k", plot_these[p], data=df)
            plt.title(call_these[p])
            plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2)
            plt.xlabel("k")
        plt.show()

    def aggregate_loocv(self):
        A = self.all_clus_labels
        A = A[self.train_index_sub, :, :]
        A = A[:, :, self.train_index_sub]
        (
            self.cr,
            self.pac,
            self.cocluster_ratio,
            self.uniqueness_pct,
        ) = n_clus_retrieval_grid(A)
        self.counts = [
            len(np.where(self.cr == i)[0])
            for i in range(np.max(self.cr).astype(int) + 1)
        ]

    def plot_cr_per_k(self):
        fig = plt.figure()
        for nk in range(self.nk):
            plt.subplot(2, 4, nk + 1)
            plt.imshow(self.cr[:, :, nk], vmin=np.min(self.cr), vmax=np.max(self.cr))
            plt.colorbar()
            plt.title(str(nk + 2) + " clusters")
            plt.xlabel("co-cluster cut-off")
            plt.xticks(np.arange(5), np.linspace(0.5, 1, 5))
            plt.ylabel("cluster uniqueness cut-off")
            plt.yticks(np.arange(5), np.linspace(0, 50, 5))
        plt.show()

    def plot_bic_pac_cr(self):
        fig = plt.figure()
        plot_these = ["bic", "pac", "counts"]
        call_these = ["BIC", "PAC", "Frequency of k unique clusters"]
        for p in range(len(plot_these)):
            plt.subplot(1, 3, p + 1)
            if p == 0:
                plt.plot(np.nanmean(eval("self." + plot_these[p]), axis=0))
            else:
                plt.plot(eval("self." + plot_these[p]))
            plt.title(call_these[p])
            if p < 2:
                plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2)
            else:
                plt.xticks(np.arange(self.nk), np.arange(self.nk) + 1)
            plt.xlabel("k")
        plt.show()

    def calc_consensus_matrix(self):
        self.consensus_matrix = np.full(
            [self.nk, len(self.train_index_sub), len(self.train_index_sub)], np.nan
        )
        self.rand_all = np.full([self.nk, len(self.train_index_sub), len(self.train_index_sub)], np.nan)

        A = self.all_clus_labels[self.train_index_sub, :, :]
        A = A[:, :, self.train_index_sub]
        for nclus in range(self.nk):
            self.consensus_matrix[nclus, :, :] = get_co_cluster_count(A[:, nclus, :])
            for a1 in range(A.shape[0]):
                for a2 in range(A.shape[0]):
                    if a2 > a1:
                        self.rand_all[nclus, a1, a2] = rand_score_withnans(A[a1, nclus,:], A[a2,nclus, :])

    def cophenetic_correlation(self):
        self.coph = np.full([self.nk], np.nan)
        for nclus in range(self.nk):
            self.coph[nclus] = coph_cor(self.consensus_matrix[nclus, :, :])

    def cluster_ensembles(self):
        self.cluster_ensembles_labels = np.full(
            [len(self.train_index_sub), self.nk], np.nan
        )
        self.randall_cluster_ensembles = np.full([self.nk, len(self.train_index_sub)], np.nan)
        self.silhouette_cluster_ensembles = np.full([self.nk], np.nan)

        clusruns = self.all_clus_labels[self.train_index_sub, :, :]
        clusruns = clusruns[:, :, self.train_index_sub]

        for nclus in range(self.nk):
            self.cluster_ensembles_labels[:, nclus] = CE.cluster_ensembles(
                clusruns[:, nclus, :], verbose=True, N_clusters_max=nclus + 2
            )

            self.randall_cluster_ensembles[nclus,:] = [
                rand_score_withnans(
                    self.cluster_ensembles_labels[:, nclus], clusruns[i, nclus, :]
                )
                for i in range(clusruns.shape[0])
            ]

            self.silhouette_cluster_ensembles[nclus] = sklearn.metrics.silhouette_score(
                self.data[self.train_index_sub, :],
                self.cluster_ensembles_labels[:, nclus],
            )

    def consensus_labels_LJ(self):
        self.consensus_labels = np.full([len(self.train_index_sub), self.nk], np.nan)
        self.randall_consensus_labels = np.full([self.nk, len(self.train_index_sub)], np.nan)
        self.silhouette_consensus_labels = np.full([self.nk], np.nan)

        clusruns = self.all_clus_labels[self.train_index_sub, :, :]
        clusruns = clusruns[:, :, self.train_index_sub]

        for nclus in range(self.nk):
            self.consensus_labels[:, nclus] = get_consensus_labels(
                clusruns[:, nclus, :], nclus + 2, 0
            )

            self.randall_consensus_labels[nclus,:] = [
                rand_score_withnans(
                    self.consensus_labels[:, nclus], clusruns[i, nclus, :]
                )
                for i in range(clusruns.shape[0])
            ]

            self.silhouette_consensus_labels[nclus] = sklearn.metrics.silhouette_score(
                self.data[self.train_index_sub, :],
                self.consensus_labels[:, nclus],
            )

    def cluster_ensembles_match_betas(self):
        self.iteration_assignments = []
        A=self.all_clus_labels[self.train_index_sub,:,:]
        A=A[:,:,self.train_index_sub]
        for nclus in range(self.nk):
            a = infer_iteration_clusmatch(self.cluster_ensembles_labels[:,nclus],A[:,nclus,:])
            self.iteration_assignments.append(a)

            bb = self.betas[self.train_index_sub, :, :, :]
            bb = bb[:, nclus, :, :]
            bb = bb[:, :nclus+2, :]