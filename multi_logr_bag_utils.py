import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from random import randrange


# function for carrying out the regression
def log_in_CV(x2use, y2use, nclus):
    # we'll be making a classifier for each group separately
    allbetas = np.zeros(
        shape=[x2use.shape[1], nclus]
    )  # pre-allocate the array because we'll be filling it step by step
    u = set(y2use)  # the unique values for y, i.e. the groups we have
    ctr = (
        -1
    )  # we're starting a counter because the groups might be [2,3,4] but we want to populate the array as [0,1,2]
    for n in u:  # loop over the groups
        ctr += 1
        y = np.zeros(shape=[y2use.shape[0]])
        y[np.where(y2use == n)] = 1
        # now there are just 2 groups: those in my target group and those outside my target group
        clf = LogisticRegression(random_state=0, solver="lbfgs")
        try:
            res = clf.fit(x2use, y)
            #        try:
            allbetas[
            :, n.astype(int)
            ] = (
                clf.coef_
            )  # n instead of ctr because i want to maintain the correct order even if a class is missing in this subset of the data
        #        except:
        #            print(Yboot.shape)
        #            print(np.sum(Yboot))
        except:
            print("only one class for " + str(nclus))
    return allbetas


def subsample(dataset, ratio=1.0):
    totalN = dataset.shape[0]
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(set(sample)) < n_sample:
        index = randrange(len(dataset))
        sample.append(index)
    while len(sample) < totalN:
        index = randrange(len(sample))
        sample.append(sample[index])
    return sample


# add bootstrapping to the function that carries out the regression
def bag_log(x2use, y2use, nboot, nclus):
    allbetas = np.zeros(shape=[x2use.shape[1], nclus, nboot])  # pre-allocate
    for i in range(nboot):
        # get and select the bootstrap sample
        bootidc = subsample(x2use, 0.666)
        Xboot = x2use[bootidc, :]
        Yboot = y2use[bootidc]
        # run it through the regression and save regression weights
        allbetas[:, :, i] = log_in_CV(Xboot, Yboot, nclus)
    # average regression weight from all bootstrap runs
    betas = np.nanmean(allbetas, axis=2)
    return betas  # and spit them out again at the end


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
