#!/usr/bin/env python
# coding: utf-8

####################################################
# Multi-class classification                       #
# - with cross-validation                          #
# - and bootstrap aggregation                      #
# in python, using sklearn                         #
#                                                  #
# Author: Lee Jollans (lee_jollans@psych.mpg.de)   #
####################################################

# module imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold

from multi_logr_bag_utils import bag_log, log_in_CV, get_metrics


def multi_logr_bagr(nboot, xy, n_groups, n_cv_folds, print_output):
    y = xy[:, 0]
    x = xy[:, 1:]

    kf = KFold(n_splits=n_cv_folds)

    # pre-allocate arrays
    overall_prediction_discrete = np.full([len(y), n_groups], np.nan)
    overall_prediction_continuous = np.full([len(y), n_groups], np.nan)
    overall_prediction_1 = np.full([len(y), n_groups], np.nan)
    auc_partial = np.full([n_cv_folds, n_groups], np.nan)
    f1_partial = np.full([n_cv_folds, n_groups], np.nan)
    betas = np.full([x.shape[1], n_groups, n_cv_folds], np.nan)
    auc = np.full([len(set(y))], np.nan)
    betas2use = np.nanmean(betas, axis=2)
    f1 = np.full([len(set(y))], np.nan)

    fold = -1
    for train_index, test_index in kf.split(x):
        fold += 1
        x_train = x[train_index, :]
        y_train = y[train_index]
        x_test = x[test_index, :]
        y_test = y[test_index]

        if nboot > 1:
            betas[:, :, fold] = bag_log(x_train, y_train, nboot, n_groups)
        else:
            betas[:, :, fold] = log_in_CV(x_train, y_train, n_groups)

        u = set(y_test)
        for n in u:
            dummy_assignment = np.zeros(shape=[len(y_test)])
            dummy_assignment[np.where(y_test == n)[0]] = 1
            [
                auc_partial[fold, n.astype(int)],
                f1_partial[fold, n.astype(int)],
                overall_prediction_discrete[test_index, n.astype(int)],
                overall_prediction_continuous[test_index, n.astype(int)],
            ] = get_metrics(x_test, dummy_assignment, betas[:, n.astype(int), fold])

    for n in range(n_groups):
        truth = np.zeros(shape=[len(y)])
        truth[np.where(y == n)[0]] = 1
        groups_in_prediction=len(set(overall_prediction_discrete[:, n]))
        if groups_in_prediction > 1:
            auc[n] = roc_auc_score(truth, overall_prediction_discrete[:, n])
            f1[n] = f1_score(truth, overall_prediction_discrete[:, n])
        else:
            print('WARNING! there were ' + str(len(set(overall_prediction_discrete[:,n]))) + ' groups predicted with truth having ' + str(len(set(truth))) + ' groups.')
        if print_output == 1:
            print("group " + str(n) + ": AUC=" + str(auc) + ", F1 score=" + str(f1))

    groupclass = np.zeros(shape=[x.shape[0]])
    for n in range(len(set(y))):
        groupclass[
            np.where(
                np.max(
                    overall_prediction_1 - np.expand_dims(overall_prediction_1[:, n], axis=1), axis=1
                )
                == 0
            )
        ] = n
    # plot overall prediction
    correctclass = np.zeros(shape=[x.shape[0]])
    correctclass[np.where((y - groupclass) == 0)[0]] = 1
    if print_output == 1:
        fig = plt.figure(figsize=[20, 4])
        fig.add_subplot(1, 3, 1)
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.title("Actual groups")
        fig.add_subplot(1, 3, 2)
        plt.scatter(x[:, 0], x[:, 1], c=groupclass)
        plt.title("Predicted groups")
        fig.add_subplot(1, 3, 3)
        plt.scatter(x[:, 0], x[:, 1], c=correctclass)
        plt.title("Correct classifications")
        plt.show()

    return (
        auc,
        f1,
        betas2use,
        overall_prediction_continuous,
        overall_prediction_discrete,
        auc_partial,
        f1_partial,
        betas,
        groupclass,
        correctclass,
        problemrec,
    )





