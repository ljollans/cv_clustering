import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from random import randrange


def bag_log(x, y, n_bags, n_groups):
    all_betas = np.zeros(shape=[x.shape[1], n_groups, n_bags])
    for i in range(n_bags):
        bootstrapping_idx = subsample(x, 0.666)
        x_bootstrapped = x[bootstrapping_idx, :]
        y_bootstrapped = y[bootstrapping_idx]
        all_betas[:, :, i] = log_in_CV(x_bootstrapped, y_bootstrapped, n_groups)
    betas = np.nanmean(all_betas, axis=2)
    return betas


def log_in_CV(x, y, n_groups):
    all_betas = np.zeros(shape=[x.shape[1], n_groups])
    u = set(y)
    ctr = -1
    for n in u:
        ctr += 1
        dummy_coded_y = np.zeros(shape=[y.shape[0]])
        dummy_coded_y[np.where(y == n)] = 1
        n_groups_in_dummy_y = len(set(dummy_coded_y))
        if n_groups_in_dummy_y>1:
            clf = LogisticRegression(random_state=0, solver="lbfgs")
            res = clf.fit(x, dummy_coded_y)
            all_betas[:, n.astype(int)] = (clf.coef_)
        else:
            print("only one class for " + str(n_groups))
    return all_betas


def subsample(x, ratio=1.0):
    n = x.shape[0]
    sample = list()
    n_sample = round(len(x) * ratio)
    while len(set(sample)) < n_sample:
        index = randrange(len(x))
        sample.append(index)
    while len(sample) < n:
        index = randrange(len(sample))
        sample.append(sample[index])
    return sample


def get_metrics(x, y, betas2use):
    prediction_continuous = np.squeeze(x.dot(betas2use))
    prediction_discrete = np.zeros(shape=[prediction_continuous.shape[0]])

    try:
        fpr, tpr, thresholds = roc_curve(y, prediction_continuous)
        max_point = np.argmax(tpr - fpr)
        prediction_discrete[np.where(prediction_continuous >= thresholds[max_point])] = 1
        auc = roc_auc_score(y, prediction_discrete)
        f1 = f1_score(y, prediction_discrete)
    except IndexError:
        print("could not calculate ROC")
        auc = np.nan
        f1 = np.nan

    return auc, f1, prediction_discrete, prediction_continuous

