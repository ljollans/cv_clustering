import numpy as np
import pickle
import csv
from cv_clustering.beta_aggregate import aggregate, get_proba
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')
from sklearn.cluster import AgglomerativeClustering


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


def agglom(input_filedir, modstr, sets, n_k, n_cv_folds):

    nclus_agglom=np.full([len(sets),n_cv_folds,n_cv_folds,n_k],np.nan)

    for s in range(len(sets)):
        print(sets[s])
        filestr = (input_filedir + sets[s] + modstr + str(0))
        with open(filestr, "rb") as f:
            mod = pickle.load(f)

        for mf in range(n_cv_folds):
            for k in range(n_k):

                X=np.empty((mod.data.shape[1],0),int)
                for sf in range(n_cv_folds):
                    fold=(mf*n_cv_folds)+sf
                    filestr=(input_filedir + sets[s] + modstr + str(fold))
                    with open(filestr, "rb") as f:
                        mod = pickle.load(f)

                    if k==0:
                        crit=np.nanmean(mod.allbetas[k],axis=1)
                        crit=np.array([crit,-crit]).T
                    else:
                        crit=np.nanmean(mod.allbetas[k],axis=2)

                    X = np.append(X,crit, axis=1)

                clustering = AgglomerativeClustering(compute_full_tree=True, distance_threshold=3, n_clusters=None,linkage='complete').fit(X.T)
                nclus_agglom[s,mf,sf,k]=clustering.n_clusters_

                clustering = AgglomerativeClustering(n_clusters=k+2, linkage='complete').fit(X.T)

                assig=np.full([n_cv_folds,k+2],np.nan)
                allbetas = np.full([X.shape[0], k + 2], np.nan)
                ctr=0
                for f in range(n_cv_folds):
                    for cc in range(k+2):
                        assig[f,cc]=clustering.labels_[ctr]
                        ctr+=1
                for cc in range(k+2):
                    allbetas[:,cc]=np.nanmean(X[:,np.where(clustering.labels_==cc)[0]], axis=1)

                with open((input_filedir + sets[s] + 'aggr_betas.pkl'), 'wb') as f:
                    pickle.dump([X,clustering,assig,allbetas],f)

    with open((input_filedir + 'nclus_agglom.pkl'), 'wb') as f:
        pickle.dump(nclus_agglom, f)