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
    auc_per_cv_fold = np.full([n_cv_folds, n_groups], np.nan)
    f1_per_cv_fold = np.full([n_cv_folds, n_groups], np.nan)
    betas_per_fold = np.full([x.shape[1], n_groups, n_cv_folds], np.nan)
    auc_across_cv_folds = np.full([len(set(y))], np.nan)
    f1_across_cv_folds = np.full([len(set(y))], np.nan)

    fold = -1
    for train_index, test_index in kf.split(x):
        fold += 1
        x_train = x[train_index, :]
        y_train = y[train_index]
        x_test = x[test_index, :]
        y_test = y[test_index]

        if nboot > 1:
            betas_per_fold[:, :, fold] = bag_log(x_train, y_train, nboot, n_groups)
        else:
            betas_per_fold[:, :, fold] = log_in_CV(x_train, y_train, n_groups)

        u = set(y_test)
        for n in u:
            dummy_assignment = np.zeros(shape=[len(y_test)])
            dummy_assignment[np.where(y_test == n)[0]] = 1
            [
                auc_per_cv_fold[fold, n.astype(int)],
                f1_per_cv_fold[fold, n.astype(int)],
                overall_prediction_discrete[test_index, n.astype(int)],
                overall_prediction_continuous[test_index, n.astype(int)],
            ] = get_metrics(x_test, dummy_assignment, betas_per_fold[:, n.astype(int), fold])

    for n in range(n_groups):
        truth = np.zeros(shape=[len(y)])
        truth[np.where(y == n)[0]] = 1
        groups_in_prediction=len(set(overall_prediction_discrete[:, n]))
        if groups_in_prediction > 1:
            try:
                auc_across_cv_folds[n] = roc_auc_score(truth, overall_prediction_discrete[:, n])
                f1_across_cv_folds[n] = f1_score(truth, overall_prediction_discrete[:, n])
            except:
                print(overall_prediction_discrete[:, n])
        else:
            print('WARNING! there were ' + str(len(set(overall_prediction_discrete[:,n]))) + ' groups predicted with truth having ' + str(len(set(truth))) + ' groups.')
        if print_output == 1:
            print("group " + str(n) + ": AUC=" + str(auc_across_cv_folds) + ", F1 score=" + str(f1_across_cv_folds))

    beta_avg_across_folds = np.nanmean(betas_per_fold, axis=2)

    return (
        auc_across_cv_folds,
        f1_across_cv_folds,
        beta_avg_across_folds,
        overall_prediction_continuous,
        overall_prediction_discrete,
        auc_per_cv_fold,
        f1_per_cv_fold,
        betas_per_fold,
    )




