#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import AgglomerativeClustering
from looco_loop import loocv_loop
from loocv_assigmatcher_nov import (
    n_clus_retrieval_chk,
    n_clus_retrieval_grid,
    get_co_cluster_count,
    get_consensus_labels,
    infer_iteration_clusmatch, collect_betas_for_corresponding_clus, sort_into_clusters_argmax_ecdf)
from utils import coph_cor, rand_score_withnans, get_gradient_change, silhouette_score_withnans, get_pac, \
    linear_multiclass_calibrated
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

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

        self.test_index_sub = np.where(
            (self.cv_assignment[:, self.mainfold] == self.subfold)
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
                self.betas[self.train_index_sub, nclus, :, :nclus + 2],
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
        A = self.all_clus_labels[self.train_index_sub, :, :]
        A = A[:, :, self.train_index_sub]
        for nclus in range(self.nk):
            self.consensus_matrix[nclus, :, :] = get_co_cluster_count(A[:, nclus, :])

    def calc_rand_all(self):
        self.rand_all = np.full([self.nk, len(self.train_index_sub), len(self.train_index_sub)], np.nan)
        A = self.all_clus_labels[self.train_index_sub, :, :]
        A = A[:, :, self.train_index_sub]
        for nclus in range(self.nk):
            for a1 in range(A.shape[0]):
                for a2 in range(A.shape[0]):
                    if a2 > a1:
                        self.rand_all[nclus, a1, a2] = rand_score_withnans(A[a1, nclus, :], A[a2, nclus, :])

    def get_pac(self):
        self.pac = np.full([self.nk], np.nan)
        for nclus in range(self.nk):
            self.pac[nclus] = get_pac(self.consensus_matrix[nclus, :, :])

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

            self.randall_cluster_ensembles[nclus, :] = [
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

            self.randall_consensus_labels[nclus, :] = [
                rand_score_withnans(
                    self.consensus_labels[:, nclus], clusruns[i, nclus, :]
                )
                for i in range(clusruns.shape[0])
            ]

            self.silhouette_consensus_labels[nclus] = sklearn.metrics.silhouette_score(
                self.data[self.train_index_sub, :],
                self.consensus_labels[:, nclus],
            )

    def hierarchical_consensus(self):
        self.agglom_labels = np.full([len(self.train_index_sub), self.nk], np.nan)
        self.randall_agglom = np.full([self.nk, len(self.train_index_sub)], np.nan)
        self.silhouette_agglom = np.full([self.nk], np.nan)

        A = self.all_clus_labels[self.train_index_sub, :, :]
        A = A[:, :, self.train_index_sub]
        for nclus in range(self.nk):
            aa = A[:, nclus, :].T
            np.fill_diagonal(aa, -1)
            clustering = AgglomerativeClustering(n_clusters=nclus + 2).fit(aa)
            self.agglom_labels[:, nclus] = clustering.labels_

            self.randall_agglom[nclus, :] = [
                rand_score_withnans(
                    self.agglom_labels[:, nclus], aa[:, i]
                )
                for i in range(len(self.train_index_sub))
            ]

            self.silhouette_agglom[nclus] = sklearn.metrics.silhouette_score(
                self.data[self.train_index_sub, :],
                self.agglom_labels[:, nclus],
            )

    def cluster_ensembles_match_betas(self):
        self.iteration_assignments = []
        self.beta_aggregate = []
        self.beta_pre_aggregate = []
        A = self.all_clus_labels[self.train_index_sub, :, :]
        A = A[:, :, self.train_index_sub]
        for nclus in range(self.nk):
            assignments = infer_iteration_clusmatch(self.cluster_ensembles_labels[:, nclus], A[:, nclus, :])
            self.iteration_assignments.append(assignments)

            betas = self.betas[self.train_index_sub, nclus, :nclus + 2, :]
            aggregated_betas, new_betas_array = collect_betas_for_corresponding_clus(assignments, betas)
            self.beta_aggregate.append(aggregated_betas)
            self.beta_pre_aggregate.append(new_betas_array)

    def cluster_ensembles_new_classification(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=None)
        allbetas = []
        allitcpt = []
        X = self.data[self.train_index_sub,:]
        y = self.cluster_ensembles_labels
        for nclus in range(self.nk):
            if nclus==0:
                tmpbetas = np.full([self.data.shape[1],  5], np.nan)
            else:
                tmpbetas = np.full([self.data.shape[1], nclus+2,5], np.nan)
            tmpitcpt = np.full([nclus+2,5], np.nan)
            fold=-1
            for train_index, test_index in kf.split(X):
                fold+=1
                clf = LogisticRegression(random_state=0, multi_class='ovr').fit(X[train_index, :],
                                                                                y[train_index, nclus],
                                                                                )
                if nclus==0:
                    tmpbetas[:,fold]= np.squeeze(clf.coef_.T)
                else:
                    tmpbetas[:, :, fold] = clf.coef_.T
                tmpitcpt[:, fold] = clf.intercept_
            allbetas.append(tmpbetas)
            allitcpt.append(tmpitcpt)
        self.allbetas = allbetas
        self.allitcpt = allitcpt


    def best_k_plot(self):
        self.test_index_sub = np.where(
            (self.cv_assignment[:, self.mainfold] == self.subfold)
            & (~np.isnan(self.cv_assignment[:, self.mainfold]))
        )[0]
        sil2 = [
            sklearn.metrics.silhouette_score(self.data[self.train_index_sub, :], self.cluster_ensembles_labels[:, i])
            for i in range(self.nk)]
        sil3 = np.full(self.nk, np.nan)
        sil4 = np.full(self.nk, np.nan)
        for k in range(self.nk):
            argmax_assig, all_ys, Y = sort_into_clusters_argmax_ecdf(self.data[self.train_index_sub],
                                                                     self.beta_aggregate[k], 1 - 1 / (k + 2))
            sil3[k] = silhouette_score_withnans(self.data[self.train_index_sub, :], argmax_assig)
            argmax_assig, all_ys, Y = sort_into_clusters_argmax_ecdf(self.data[self.test_index_sub],
                                                                     self.beta_aggregate[k], 1 - 1 / (k + 2))
            sil4[k] = silhouette_score_withnans(self.data[self.test_index_sub, :], argmax_assig)

        fig = plt.figure()
        plt.subplot(2, 3, 1);
        plt.plot(np.nanmean(self.bic, axis=0));
        plt.title('Average BIC for LOOCV iterations')
        plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2);
        plt.xlabel("k")

        plt.subplot(2, 3, 2);
        plt.plot(np.nanmean(self.sil, axis=0));
        plt.plot(sil2);
        plt.plot(sil3);
        plt.plot(sil4);
        plt.title('Silhouette score');
        plt.legend(['mean for i LOOCV', 'from consensus labels', 'from beta argmax train', 'from beta argmax test'])
        plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2);
        plt.xlabel("k")

        plt.subplot(2, 3, 3);
        plt.plot(np.nanmean(np.nanmean(self.rand_all, axis=2), axis=1));
        plt.title('Average rand score btw LOOCV iterations')
        plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2);
        plt.xlabel("k")

        plt.subplot(2, 3, 4);
        plt.plot(self.pac);
        plt.plot(get_gradient_change(self.pac));
        plt.title('Proportion ambiguous clustering');
        plt.legend(['pac', 'pac gradient change'])
        plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2);
        plt.xlabel("k")

        plt.subplot(2, 3, 5);
        plt.plot(self.coph);
        plt.title('Cophenetic correlation');
        plt.xticks(np.arange(self.nk), np.arange(self.nk) + 2);
        plt.xlabel("k")

        plt.show()


def extract_vals(filedir, sets, topull, nk, ncv, withctr, save):
    if withctr == 1:
        all_tmp = np.full([nk, len(sets), 2, ncv, ncv], np.nan)
    else:
        all_tmp = np.full([nk, len(sets), ncv, ncv], np.nan)

    for s in range(len(sets)):
        if withctr == 1:
            for ctr in range(2):
                if ctr == 0:
                    pkl_filename = filedir + sets[s] + '__' + topull + '.pkl'
                else:
                    pkl_filename = filedir + sets[s] + '_ctrl__' + topull + '.pkl'

                with open(pkl_filename, "rb") as file:
                    tmp = pickle.load(file)
                fold = -1
                for mf in range(ncv):
                    for sf in range(ncv):
                        fold += 1
                        if tmp[fold].shape[0] == nk:
                            if len(tmp[fold].shape) == 1:
                                all_tmp[:, s, ctr, mf, sf] = tmp[fold]
                            elif len(tmp[fold].shape) == 2:
                                all_tmp[:, s, ctr, mf, sf] = [np.nanmean(tmp[fold][i, :]) for i in range(nk)]
                            elif len(tmp[fold].shape) == 3:
                                all_tmp[:, s, ctr, mf, sf] = [np.nanmean(tmp[fold][i, :, :]) for i in range(nk)]
                            else:
                                print('odd shape:', tmp.shape)
                        else:
                            print('odd shape:', tmp.shape)
        else:
            pkl_filename = filedir + sets[s] + '__' + topull + '.pkl'
            with open(pkl_filename, "rb") as file:
                tmp = pickle.load(file)
            fold = -1
            for mf in range(ncv):
                for sf in range(ncv):
                    fold += 1
                    if tmp[fold].shape[0] == nk:
                        if len(tmp[fold].shape) == 1:
                            all_tmp[:, s, mf, sf] = tmp[fold]
                        elif len(tmp[fold].shape) == 2:
                            all_tmp[:, s, mf, sf] = [np.nanmean(tmp[fold][i, :]) for i in range(nk)]
                        elif len(tmp[fold].shape) == 3:
                            all_tmp[:, s, mf, sf] = [np.nanmean(tmp[fold][i, :, :]) for i in range(nk)]
                        else:
                            print('odd shape:', tmp.shape)
                    else:
                        print('odd shape:', tmp.shape)

        print('set: ' + sets[s] + ':')
        print(np.nanmean(np.nanmean(all_tmp[:, s, :, :, :], axis=3), axis=2))

    if save == 1:
        with open('all_' + topull + '.pkl', "wb") as file:
            pickle.dump(all_tmp, file)

    if withctr == 1:
        df = pd.DataFrame({}, columns=[topull, 'set', 'ctr', 'mf', 'sf', 'k'])
        for s in range(len(sets)):
            for ctr in range(2):
                for mf in range(ncv):
                    for sf in range(ncv):
                        for k in range(nk):
                            tmp_df = pd.DataFrame(
                                {topull: [all_tmp[k, s, ctr, mf, sf]],
                                 'set': sets[s],
                                 'ctr': ctr,
                                 'mf': mf,
                                 'sf': sf,
                                 'k': int(k) + 2},
                                columns=[topull, 'set', 'ctr', 'mf', 'sf', 'k'])
                            df = df.append(tmp_df)
    else:
        df = pd.DataFrame({}, columns=[topull, 'set', 'mf', 'sf', 'k'])
        for s in range(len(sets)):
            for mf in range(ncv):
                for sf in range(ncv):
                    for k in range(nk):
                        _

    return all_tmp, df

class kcluster:
    def __init__(self, mod,k):

        self.k = k+2
        self.cv_assignment = mod.cv_assignment
        self.mainfold = mod.mainfold
        self.subfold = mod.subfold
        self.train_index_sub = mod.train_index_sub
        self.test_index_sub = mod.test_index_sub
        self.allbetas = mod.allbetas[k]
        self.allitcpt = mod.allitcpt[k]


    def decision_function(self, X):
        meanbetas = np.nanmean(self.allbetas,axis=2)
        meanitcpt = np.nanmean(self.itcpt, axis=1)
        ypred = X.dot(meanbetas)+meanitcpt
        return ypred
