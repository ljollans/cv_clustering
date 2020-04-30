import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


def subfold_mse(all_subfold_betas):
    # all_subfold_betas is a list of length n_cv
    # each entry has the shape nfeatures x nclusters
    ncv = len(all_subfold_betas)
    nclus = all_subfold_betas[0].shape[1]

    allmses = np.full([ncv, ncv, nclus, nclus], np.nan)
    allranks = np.full([ncv, ncv, nclus, nclus], np.nan)
    mutual_best_fit = np.full([ncv, ncv], np.nan)
    mutual_best_fit_mse = np.full([ncv, ncv], np.nan)
    n_unique_fits = np.full([ncv, ncv], np.nan)
    matched_fit_mse = np.full([ncv, ncv], np.nan)

    for sf1 in range(ncv):
        for sf2 in range(ncv):
            if sf1 != sf2:
                for clus1 in range(nclus):
                    for clus2 in range(nclus):
                        allmses[sf1, sf2, clus1, clus2] = vector_mse(all_subfold_betas[sf1][:, clus1],
                                                                     all_subfold_betas[sf2][:, clus2])
                    allranks[sf1, sf2, clus1, :] = np.argsort(allmses[sf1, sf2, clus1, :])
            z = np.where(allranks[sf1, sf2, :, :] == 0)
            n_unique_fits[sf1, sf2] = len(np.unique(z[1]))
            matched_fit_mse[sf1, sf2] = np.nanmean(
                [allmses[sf1, sf2, z[0][i], z[1][i]] for i in range(len(z[0]))])

    for sf1 in range(ncv):
        for sf2 in range(ncv):
            if sf1 != sf2:
                mfit = np.where(allranks[sf1, sf2, :, :] + allranks[sf2, sf1, :, :] == 0)
                mutual_best_fit[sf1, sf2] = len(mfit[0])
                mutual_best_fit_mse[sf1, sf2] = np.nanmean(
                    [allmses[sf1, sf2, mfit[0][i], mfit[1][i]] for i in range(len(mfit[0]))])

    return allmses, allranks, n_unique_fits, mutual_best_fit, mutual_best_fit_mse, matched_fit_mse


def vector_mse(a, b):
    scaler = StandardScaler()
    anan = np.where(np.isnan(a))[0]
    bnan = np.where(np.isnan(b))[0]
    if len(anan)>0:
        print('Warning: ' + str(len(anan)) + ' NaNs in first vector')
    if len(bnan)>0:
        print('Warning: ' + str(len(bnan)) + ' NaNs in second vector')
    if (len(anan) < (len(a)/20)) or (len(bnan) < (len(b)/20)):
        a = np.delete(a, np.where(np.isnan(b))[0])
        b = np.delete(b, np.where(np.isnan(b))[0])
        b = np.delete(b, np.where(np.isnan(a))[0])
        a = np.delete(a, np.where(np.isnan(a))[0])

        vmse = metrics.mean_squared_error(scaler.fit_transform(np.expand_dims(a, axis=1)),
                                          scaler.fit_transform(np.expand_dims(b, axis=1)))
    else:
        vmse = np.nan

    return vmse


def combine_weighted_betas(allmses, all_subfold_betas, idx_fold):
    nclus = all_subfold_betas[0].shape[1]
    nfeatures = all_subfold_betas[0].shape[0]
    ncv = len(all_subfold_betas)
    all_weighted_betas = np.full([nclus, nfeatures, ncv], np.nan)
    aggregated_betas = np.full([nfeatures, nclus], np.nan)

    for k in range(nclus):
        index_beta = all_subfold_betas[idx_fold][:, k]
        all_weighted_betas[k, :, idx_fold] = index_beta

        for cv in range(ncv):
            if cv != idx_fold:
                all_weighted_cvbetas = np.full([nfeatures, nclus], np.nan)

                for clus1 in range(nclus):
                    crit = all_subfold_betas[cv][:, clus1]
                    weight = (1 / (1 + allmses[idx_fold, cv, k, clus1]))
                    all_weighted_cvbetas[:, clus1] = crit * weight

                all_weighted_betas[k, :, cv] = np.nanmean(all_weighted_cvbetas, axis=1)
        aggregated_betas[:, k] = np.nanmean(all_weighted_betas[k, :, :], axis=1)
    return all_weighted_betas, aggregated_betas


def aggregate(all_subfold_betas):
    allmses, allranks, n_unique_fits, mutual_best_fit, mutual_best_fit_mse, matched_fit_mse = subfold_mse(
        all_subfold_betas)
    bilateral_matched_fit_error = np.nansum(matched_fit_mse, axis=0) + np.nansum(matched_fit_mse, axis=1)
    smallest_overall_error = np.where(bilateral_matched_fit_error == np.nanmin(bilateral_matched_fit_error))[0]
    all_weighted_betas, aggregated_betas = combine_weighted_betas(allmses, all_subfold_betas, smallest_overall_error[0])
    return aggregated_betas, all_weighted_betas


def get_proba(Xtrain, labels, betas, Xtest):
    # here i am assuming that the first row of betas is the intercept row
    itcpt = betas[0, :]
    betas = betas[1:, :]
    k = betas.shape[1]
    clf = LogisticRegression(random_state=0, multi_class='ovr').fit(Xtrain, labels)
    clf.intercept_ = itcpt
    clf.coef_ = betas
    clf_isotonic = CalibratedClassifierCV(clf, method='sigmoid').fit(Xtrain, labels)
    train_proba = clf_isotonic.predict_proba(Xtrain)
    test_proba = clf_isotonic.predict_proba(Xtest)
    return train_proba, test_proba

def predictargmax(X, betas):
    tmpX = np.append(np.ones(shape=[X.shape[0], 1]), X, axis=1)
    newY = tmpX.dot(betas)
    argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
    return argmaxY

