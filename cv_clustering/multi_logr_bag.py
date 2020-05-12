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
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from cv_clustering.multi_logr_bag_utils import bag_log, log_in_CV, get_metrics


def multi_logr_bagr(nboot, yx, n_groups, n_cv_folds, print_output):
    y = yx[:, 0]
    x = yx[:, 1:]

    kf = KFold(n_splits=n_cv_folds)

    # pre-allocate arrays
    try:
        n_unique_in_y=len(np.unique(y.astype(int)))
    except:
        n_unique_in_y=len(set(y))
    overall_prediction_discrete = np.full([len(y), n_groups], np.nan)
    overall_prediction_continuous = np.full([len(y), n_groups], np.nan)
    auc_per_cv_fold = np.full([n_cv_folds, n_groups], np.nan)
    f1_per_cv_fold = np.full([n_cv_folds, n_groups], np.nan)
    betas_per_fold = np.full([x.shape[1], n_groups, n_cv_folds], np.nan)
    auc_across_cv_folds = np.full([n_unique_in_y], np.nan)
    f1_across_cv_folds = np.full([n_unique_in_y], np.nan)
    n_across_cv_folds = np.full([n_unique_in_y], np.nan)

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
            prediction = overall_prediction_discrete[:,n]
            truth=np.delete(truth,np.where(np.isnan(prediction))[0])
            prediction = np.delete(prediction, np.where(np.isnan(prediction))[0])
            n_across_cv_folds[n]=len(prediction)
            auc_across_cv_folds[n] = roc_auc_score(truth, prediction)
            f1_across_cv_folds[n] = f1_score(truth, prediction)
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
        n_across_cv_folds
    )


def multi_logr_bagr(nboot, yx, n_groups, n_cv_folds, print_output):
    y = yx[:, 0]
    x = yx[:, 1:]

    kf = KFold(n_splits=n_cv_folds)

    # pre-allocate arrays
    try:
        n_unique_in_y=len(np.unique(y.astype(int)))
    except:
        n_unique_in_y=len(set(y))
    overall_prediction_discrete = np.full([len(y), n_groups], np.nan)
    overall_prediction_continuous = np.full([len(y), n_groups], np.nan)
    auc_per_cv_fold = np.full([n_cv_folds, n_groups], np.nan)
    f1_per_cv_fold = np.full([n_cv_folds, n_groups], np.nan)
    betas_per_fold = np.full([x.shape[1], n_groups, n_cv_folds], np.nan)
    auc_across_cv_folds = np.full([n_unique_in_y], np.nan)
    f1_across_cv_folds = np.full([n_unique_in_y], np.nan)
    n_across_cv_folds = np.full([n_unique_in_y], np.nan)

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
            prediction = overall_prediction_discrete[:,n]
            truth=np.delete(truth,np.where(np.isnan(prediction))[0])
            prediction = np.delete(prediction, np.where(np.isnan(prediction))[0])
            n_across_cv_folds[n]=len(prediction)
            auc_across_cv_folds[n] = roc_auc_score(truth, prediction)
            f1_across_cv_folds[n] = f1_score(truth, prediction)
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
        n_across_cv_folds
    )


def new_multiclassifCV(X, y, ncv, nbag):
    kf = KFold(n_splits=ncv, shuffle=True, random_state=None)
    allbetas = []
    allitcpt = []

    ug = np.unique(y)
    if len(ug) == 2:
        meanbetas = np.full([ncv, X.shape[1]], np.nan)
        meanitcpt = np.full([ncv], np.nan)
    else:
        meanbetas = np.full([ncv, X.shape[1], len(ug)], np.nan)
        meanitcpt = np.full([ncv, len(ug)], np.nan)

    fold = -1
    for train_index, test_index in kf.split(X):
        fold += 1

        clf = LogisticRegression(random_state=0, multi_class='ovr').fit(X[train_index, :],
                                                                        y[train_index],
                                                                        )

        if nbag == 1:
            betas2add = clf.coef_.T
            itcpt2add = clf.intercept_
        else:
            bag_regr = BaggingClassifier(base_estimator=clf, n_estimators=nbag, max_samples=0.66, max_features=1.0,
                                         bootstrap=True, bootstrap_features=False, oob_score=False,
                                         warm_start=False,
                                         n_jobs=None, random_state=None, verbose=0)
            bag_regr.fit(X[train_index, :], y[train_index])

            if len(ug) == 2:
                betas_all = np.full([X.shape[1], nbag], np.nan)
                itcpt_all = np.full([nbag], np.nan)
                for bag in range(nbag):
                    betas_all[:, bag] = bag_regr.estimators_[bag].coef_[0]
                    itcpt_all[bag] = bag_regr.estimators_[bag].intercept_
                meanbetas[fold, :] = np.nanmedian(betas_all, axis=1)
                meanitcpt[fold] = np.nanmedian(itcpt_all)
            else:
                betas_all = np.full([X.shape[1], len(ug), nbag], np.nan)
                itcpt_all = np.full([len(ug), nbag], np.nan)
                for bag in range(nbag):
                    for c in range(len(ug)):
                        betas_all[:, c, bag] = bag_regr.estimators_[bag].coef_[c]
                        itcpt_all[c, bag] = bag_regr.estimators_[bag].intercept_[c]
                meanbetas[fold, :, :] = np.nanmedian(betas_all, axis=2)
                meanitcpt[fold, :] = np.nanmedian(itcpt_all, axis=1)

    return np.nanmedian(meanbetas, axis=0), np.nanmedian(meanitcpt, axis=0)


