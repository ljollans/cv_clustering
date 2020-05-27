import numpy as np
import pickle
import csv
from cv_clustering.beta_aggregate import aggregate, get_proba
from sklearn.metrics import silhouette_samples, silhouette_score

def aggr(current_set,mainfold,k,input_files_dir, savedir, cv_assignment, method, modstr='_mod_'):
    n = cv_assignment.shape[0]
    maintrain = np.where(np.isfinite(cv_assignment[:, mainfold]))[0]
    maintest = np.where(np.isnan(cv_assignment[:, mainfold]))[0]

    data_path = input_files_dir + current_set + ".csv"
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)

    allbetas=[]
    labels = []

    for subfold in range(4):
        fold = (mainfold*4)+subfold

        pkl_filename = (savedir + current_set + modstr + str(fold))
        with open(pkl_filename, "rb") as file:
            mod = pickle.load(file)
        labels.append(mod.cluster_ensembles_labels[:,k])
        i=np.nanmean(mod.allitcpt[k], axis=1)
        if k>0:
            b=np.nanmean(mod.allbetas[k],axis=2)
        else:
            b = np.nanmean(mod.allbetas[k], axis=1)
            b = np.expand_dims(b, axis=1)
            b = np.append(b, -b, axis=1)
        allbetas.append(np.append(np.expand_dims(i,axis=1).T,b,axis=0))
    aggregated_betas, all_weighted_betas = aggregate(allbetas, method)

    Xtrain = data[maintrain, :]
    tmpX = np.append(np.ones(shape=[Xtrain.shape[0], 1]), Xtrain, axis=1)
    newY = tmpX.dot(aggregated_betas)
    argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
    tmp_trainproba, tmp_testproba = get_proba(Xtrain, argmaxY, aggregated_betas, data)


    argmaxY = np.array([np.where(tmp_testproba[i, :] == np.max(tmp_testproba[i, :]))[0][0] for i in
                        range(tmp_testproba.shape[0])])
    valmaxY = np.array([np.max(tmp_testproba[i, :]) for i in range(tmp_testproba.shape[0])])
    trainproba = valmaxY[maintrain]
    testproba = valmaxY[maintest]

    if len(np.unique(argmaxY)) > 1:
        sil = silhouette_score(data, argmaxY)
        n_sil = silhouette_samples(data, argmaxY)
    else:
        sil=np.nan
        n_sil=np.nan

    return allbetas, aggregated_betas, all_weighted_betas, argmaxY, valmaxY, trainproba, testproba, sil, n_sil