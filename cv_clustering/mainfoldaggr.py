import numpy as np
import pickle
import csv

import sklearn

from cv_clustering.beta_aggregate import aggregate, get_proba, assigfromproba, vector_mse, predictargmax
from cv_clustering.utils import contingency_matrix, rand_score_withnans
from sklearn.metrics import silhouette_samples, silhouette_score
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import pathlib
from scipy.cluster.hierarchy import dendrogram
from sklearn import preprocessing

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

    pathlib.Path(input_filedir + 'dendograms').mkdir(exist_ok=True)

    for s in range(len(sets)):
        fig=plt.figure(figsize=[30,15])

        print(sets[s])
        filestr = (input_filedir + sets[s] + modstr + str(0))
        with open(filestr, "rb") as f:
            mod = pickle.load(f)

        for mf in range(n_cv_folds):

            maintrain = np.where(np.isfinite(mod.cv_assignment[:, mf]))[0]
            maintest = np.where(np.isnan(mod.cv_assignment[:, mf]))[0]

            for k in range(n_k):

                X=np.empty((mod.data.shape[1],0),int)
                IX=np.empty((0),int)
                for sf in range(n_cv_folds):
                    fold=(mf*n_cv_folds)+sf
                    filestr=(input_filedir + sets[s] + modstr + str(fold))
                    with open(filestr, "rb") as f:
                        mod = pickle.load(f)

                    if k==0:
                        crit=np.nanmean(mod.allbetas[k],axis=1)
                        crit=np.array([crit,-crit]).T
                        criti=np.nanmean(mod.allitcpt[k])
                        criti = np.array([criti, -criti]).T
                    else:
                        crit=np.nanmean(mod.allbetas[k],axis=2)
                        criti=np.nanmean(mod.allitcpt[k],axis=1)

                    X = np.append(X,crit, axis=1)
                    IX = np.append(IX, criti)


                # for now I decided to neither scale nor include the intercept as we are just looking at patterns.
                # scaling would move what betas are positive vs. negative
                # the intercept doesn;t add any information about the pattern, just the group sizes (?)

                clustering = AgglomerativeClustering(compute_full_tree=True, distance_threshold=3, n_clusters=None,linkage='ward').fit(X.T)
                nclus_agglom[s,mf,sf,k]=clustering.n_clusters_
                # make dendogram to save
                plt.subplot(n_cv_folds,n_k,(mf*n_k)+k+1)
                plt.title('k=' + str(k+2))
                plot_dendrogram(clustering, p=3)

                clustering = AgglomerativeClustering(n_clusters=k+2, linkage='complete').fit(X.T)

                assig=np.full([n_cv_folds,k+2],np.nan)
                allbetas = np.full([X.shape[0], k + 2], np.nan)
                ctr=0
                for f in range(n_cv_folds):
                    for cc in range(k+2):
                        assig[f,cc]=clustering.labels_[ctr]
                        ctr+=1
                for cc in range(k+2):
                    allbetas[:, cc] = np.nanmean(X[:, np.where(clustering.labels_ == cc)[0]], axis=1)

                # not we have the betas -- apply and get probas
                newY1 = mod.data[maintrain, :].dot(allbetas)
                argmaxY = np.array([np.where(newY1[i, :] == np.max(newY1[i, :]))[0][0] for i in range(newY1.shape[0])])
                newYtest = mod.data[maintest, :].dot(allbetas)
                argmaxYtest = np.array([np.where(newYtest[i, :] == np.max(newYtest[i, :]))[0][0] for i in range(newYtest.shape[0])])

                tmp_trainproba, tmp_testproba = get_proba(mod.data[maintrain, :], argmaxY, allbetas, mod.data[maintest, :])
                argmaxY2 = np.array([np.where(tmp_trainproba[i, :] == np.max(tmp_trainproba[i, :]))[0][0] for i in range(tmp_trainproba.shape[0])])
                argmaxYtest2 = np.array([np.where(tmp_testproba[i, :] == np.max(tmp_testproba[i, :]))[0][0] for i in range(tmp_testproba.shape[0])])


                with open((input_filedir + sets[s] + '_aggr_betas_k' + str(k) + '_mf' + str(mf) + '.pkl'), 'wb') as f:
                    pickle.dump([X,clustering,assig,allbetas, tmp_testproba, argmaxYtest, argmaxYtest2],f)

        plt.savefig(input_filedir + 'dendograms/' + sets[s] + '_clus_dendograms.png')


    with open((input_filedir + 'nclus_agglom.pkl'), 'wb') as f:
        pickle.dump([nclus_agglom], f)




def agglom_best_k_per_sf(input_filedir, modstr, input_filedir_null, modstr_null, sets, setsize, n_cv_folds, n):

    with open(input_filedir + modstr + 'sil_f1_prob_lvl2.pkl', 'rb') as f:
        [silhouette1_lvl2, silhouette2_lvl2, microf1_lvl2, macrof1_lvl2, testproba_lvl2, testprobadist_lvl2, clussize_CE_lvl2,
         clussize_test_lvl2] = pickle.load(f)

    with open(input_filedir_null + modstr_null + 'sil_f1_prob_lvl2.pkl', 'rb') as f:
        [silhouette1_lvl2n, silhouette2_lvl2n, microf1_lvl2n, macrof1_lvl2n, testproba_lvl2n, testprobadist_lvl2, clussize_CE_lvl2n,
         clussize_test_lvl2n] = pickle.load(f)

    # get best k for each subfold based on difference from null for silhouette2_lvl2
    bestk_mdd = np.full([len(sets), n_cv_folds, n_cv_folds], np.nan)
    allbetas_sfk = [None] * len(sets)
    final_k=np.full([len(sets), n_cv_folds], np.nan)

    silmaintrain = np.full([len(sets), n_cv_folds], np.nan)
    silmaintest = np.full([len(sets), n_cv_folds], np.nan)

    train_clus_assig = np.full([n,len(sets), n_cv_folds], np.nan)
    test_clus_assig = np.full([n, len(sets), n_cv_folds], np.nan)
    train_clus_assig25 = np.full([n, len(sets), n_cv_folds], np.nan)
    test_clus_assig25 = np.full([n, len(sets), n_cv_folds], np.nan)
    train_clus_prob = np.full([n, len(sets), n_cv_folds], np.nan)
    test_clus_prob = np.full([n, len(sets), n_cv_folds], np.nan)

    for s in range(len(sets)):
        allbetas_sfk[s] = []
        print(sets[s])
        fig = plt.figure(figsize=[20, 4])
        for mf in range(n_cv_folds):
            X = np.empty((setsize[s], 0), int)

            for sf in range(n_cv_folds):

                fold = (mf * n_cv_folds) + sf
                filestr = (input_filedir + sets[s] + modstr + str(fold))
                with open(filestr, "rb") as f:
                    mod = pickle.load(f)

                # identify best k for this subfold
                act_crit = silhouette2_lvl2[s, mf, sf, :]
                null_crit = silhouette2_lvl2n[s, mf, sf, :]
                critd = act_crit - null_crit
                maxcrit = np.nanmax(critd)
                maxcritwhere = np.where(critd == maxcrit)[0]
                bestk_mdd[s, mf, sf] = maxcritwhere

                # append the betas to X
                if bestk_mdd[s, mf, sf] == 0:
                    crit = np.nanmean(mod.allbetas[bestk_mdd[s, mf, sf].astype(int)], axis=1)
                    crit = np.array([crit, -crit]).T
                    criti = np.nanmean(mod.allitcpt[bestk_mdd[s, mf, sf].astype(int)])
                    criti = np.array([criti, -criti]).T
                else:
                    crit = np.nanmean(mod.allbetas[bestk_mdd[s, mf, sf].astype(int)], axis=2)
                    criti = np.nanmean(mod.allitcpt[bestk_mdd[s, mf, sf].astype(int)], axis=1)

                X = np.append(X, crit, axis=1)

            # cluster all subfold betas
            clustering = AgglomerativeClustering(compute_full_tree=True, distance_threshold=3, n_clusters=None,
                                                 linkage='ward').fit(X.T)
            nk = len(np.unique(clustering.labels_))
            final_k[s,mf]=nk

            # plot dendogram for later reference
            plt.subplot(1, n_cv_folds, mf + 1)
            plot_dendrogram(clustering, p=3)
            plt.title((sets[s], str(mf), str(len(np.unique(clustering.labels_)))))

            # average the betas for each glob
            allbetas = np.full([X.shape[0], nk], np.nan)
            for cc in range(nk):
                allbetas[:, cc] = np.nanmean(X[:, np.where(clustering.labels_ == cc)[0]], axis=1)
            allbetas_sfk[s].append(allbetas)

            # get cluster assignments
            maintest = np.where(np.isnan(mod.cv_assignment[:, mf]))[0]
            maintrain = np.where(np.isfinite(mod.cv_assignment[:, mf]))[0]

            newY1 = mod.data[maintrain, :].dot(allbetas_sfk[s][mf])
            argmaxYtrain = np.array([np.where(newY1[i, :] == np.max(newY1[i, :]))[0][0] for i in range(newY1.shape[0])])
            tmp_trainproba, tmp_testproba = get_proba(mod.data[maintrain, :], argmaxYtrain, allbetas_sfk[s][mf],mod.data[maintest, :])
            if tmp_trainproba.shape[1] < newY1.shape[1]:
                newtmptrainproba = np.zeros(shape=[tmp_trainproba.shape[0], newY1.shape[1]])
                newtmptestproba = np.zeros(shape=[tmp_testproba.shape[0], newY1.shape[1]])
                u = -1
                for b in range(newY1.shape[1]):
                    if len(np.unique(allbetas_sfk[s][mf][:, b])) > 1:
                        u += 1
                        newtmptrainproba[:, b] = tmp_trainproba[:, u]
                        newtmptestproba[:, b] = tmp_testproba[:, u]
                tmp_trainproba = newtmptrainproba
                tmp_testproba = newtmptestproba

            assignment_train, likelihood_train, assignment25_train = assigfromproba(tmp_trainproba, 4)
            assignment_test, likelihood_test, assignment25_test = assigfromproba(tmp_testproba, 4)
            silmaintrain[s, mf] = silhouette_score(mod.data[maintrain, :], assignment_train)
            silmaintest[s, mf] = silhouette_score(mod.data[maintest, :], assignment_test)

            train_clus_assig[maintrain,s,mf] = assignment_train
            test_clus_assig[maintest,s,mf] = assignment_test
            train_clus_prob[maintrain,s,mf] = likelihood_train
            test_clus_prob[maintest,s,mf] = likelihood_test
            train_clus_assig25[maintrain,s,mf] = assignment25_train
            test_clus_assig25[maintest,s,mf] = assignment25_test

        plt.savefig(input_filedir + 'dendograms/mf_bestkaggr_' + sets[s] + '_clus_dendograms.png')
    with open((input_filedir + 'aggr_betas_best_sf_k.pkl'), 'wb') as f:
        pickle.dump([bestk_mdd,final_k,allbetas_sfk,silmaintrain,silmaintest,train_clus_assig,test_clus_assig,train_clus_prob,test_clus_prob,train_clus_assig25,test_clus_assig25], f)



def aggr4comp(input_filedir, modstr, sets, n_k, n_cv_folds):


    for s in range(len(sets)):

        print(sets[s])
        filestr = (input_filedir + sets[s] + modstr + str(0))
        with open(filestr, "rb") as f:
            mod = pickle.load(f)

        for mainfold in range(n_cv_folds):

            maintrain = np.where(np.isfinite(mod.cv_assignment[:, mainfold]))[0]
            maintest = np.where(np.isnan(mod.cv_assignment[:, mainfold]))[0]

            for k in range(n_k):

                allbetas=[]

                for subfold in range(4):
                    fold = (mainfold*4)+subfold

                    filestr = (input_filedir + sets[s] + modstr + str(fold))
                    with open(filestr, "rb") as f:
                        mod = pickle.load(f)

                    if k==0:
                        b=np.nanmean(mod.allbetas[k],axis=1)
                        b=np.array([b,-b]).T
                        i=np.nanmean(mod.allitcpt[k])
                        i = np.array([i, -i]).T
                    else:
                        b=np.nanmean(mod.allbetas[k],axis=2)
                        i=np.nanmean(mod.allitcpt[k],axis=1)

                    allbetas.append(np.append(np.expand_dims(i,axis=1).T,b,axis=0))
                aggregated_betas, all_weighted_betas = aggregate(allbetas, 0)

                # not we have the betas -- apply and get probas
                newY1 = mod.data[maintrain, :].dot(aggregated_betas[1:,:])
                argmaxY = np.array([np.where(newY1[i, :] == np.max(newY1[i, :]))[0][0] for i in range(newY1.shape[0])])
                newYtest = mod.data[maintest, :].dot(aggregated_betas[1:,:])
                argmaxYtest = np.array(
                    [np.where(newYtest[i, :] == np.max(newYtest[i, :]))[0][0] for i in range(newYtest.shape[0])])

                tmp_trainproba, tmp_testproba = get_proba(mod.data[maintrain, :], argmaxY, aggregated_betas[1:,:],
                                                          mod.data[maintest, :])
                argmaxY2 = np.array([np.where(tmp_trainproba[i, :] == np.max(tmp_trainproba[i, :]))[0][0] for i in
                                     range(tmp_trainproba.shape[0])])
                argmaxYtest2 = np.array([np.where(tmp_testproba[i, :] == np.max(tmp_testproba[i, :]))[0][0] for i in
                                         range(tmp_testproba.shape[0])])

                with open((input_filedir + sets[s] + '_aggr_00_betas_k' + str(k) + '_mf' + str(mainfold) + '.pkl'), 'wb') as f:
                    pickle.dump([aggregated_betas, all_weighted_betas, tmp_testproba, argmaxYtest, argmaxYtest2], f)





def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# stick everything into one function and do it for all sets
def doagglomchks(s, link, input_filediri, modstri, mf,n_cv_folds, sets, setsize, n_k):
    # step 1: collect all vectors across subfolds for each k
    k_collect = [None] * n_k

    for k in range(n_k):

        k_collect[k] = np.empty((setsize[s], 0), int)
        for sf in range(n_cv_folds):
            fold = (mf * n_cv_folds) + sf
            filestr = (input_filediri + sets[s] + modstri + str(fold))
            with open(filestr, "rb") as f:
                mod = pickle.load(f)

            if k == 0:
                crit = np.nanmean(mod.allbetas[k], axis=1)
                crit = np.array([crit, -crit]).T
                criti = np.nanmean(mod.allitcpt[k])
                criti = np.array([criti, -criti]).T
            else:
                crit = np.nanmean(mod.allbetas[k], axis=2)
                criti = np.nanmean(mod.allitcpt[k], axis=1)
            if len(np.where(np.isnan(crit))[0])==len(crit):
                pass
            else:
                k_collect[k] = np.append(k_collect[k], crit, axis=1)

    # step 3: force k threshold
    avgclus = [None] * n_k
    for kthresh in range(n_k):
        avgclus[kthresh] = []
        for k in range(n_k):
            if k >= kthresh:
                if link == 0:
                    clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='ward').fit(k_collect[k].T)
                elif link == 1:
                    clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='complete').fit(k_collect[k].T)
                elif link == 2:
                    clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='average').fit(k_collect[k].T)
                else:
                    print('please select linkage function')
                avgbs = np.full([k_collect[k].shape[0], kthresh + 2], np.nan)
                for c in range(kthresh + 2):
                    avgbs[:, c] = np.nanmean(k_collect[k][:, np.where(clustering.labels_ == c)[0]], axis=1)
                avgclus[kthresh].append(avgbs)

    # step 4: calculate match between each solution
    averageerror = np.zeros(n_k)
    for kthresh in range(n_k):
        bestmatch = []
        for mf1 in range(len(avgclus[kthresh])):
            for mf2 in range(len(avgclus[kthresh])):
                if mf1 != mf2:
                    allmses = np.full([kthresh + 2, kthresh + 2], np.nan)
                    for c1 in range(kthresh + 2):
                        for c2 in range(kthresh + 2):
                            a = avgclus[kthresh][mf1][:, c1]
                            b = avgclus[kthresh][mf2][:, c2]
                            allmses[c1, c2] = vector_mse(a, b)
                    for c in range(kthresh + 2):
                        a = np.where(allmses == np.nanmin(allmses))
                        bestmatch.append(np.nanmin(allmses))
                        allmses[a[0][0], :] = np.nan
                        allmses[:, a[1][0]] = np.nan
        averageerror[kthresh] = np.nanmean(np.array(bestmatch))
    return averageerror


def moreagglomloop(a):
    s=a[0]
    input_filediri=a[1]
    modstri=a[2]
    sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s",
            "Tct_Scs_tc_sc_s"]
    setsize = np.array([82, 82, 150, 84, 84, 154, 82, 82, 150, 84, 84, 154])
    mferrormat = np.full([20, 8, 4], np.nan)
    for mf in range(4):
        mferrormat[:, :, mf] = doagglomchks_more(s=s, link=0, input_filediri=input_filediri, modstri=modstri, mf=mf,
                                            n_cv_folds=4, sets=sets, setsize=setsize, n_k=8,
                                            n_threshs=20)
    return mferrormat

def doagglomchks_more(s, link, input_filediri, modstri, mf, n_cv_folds, sets, setsize, n_k, n_threshs):
    # step 1: collect all vectors across subfolds for each k
    k_collect = [None] * n_k

    for k in range(n_k):

        k_collect[k] = np.empty((setsize[s], 0), int)
        for sf in range(n_cv_folds):
            fold = (mf * n_cv_folds) + sf
            filestr = (input_filediri + sets[s] + modstri + str(fold))
            with open(filestr, "rb") as f:
                mod = pickle.load(f)

            if k == 0:
                crit = np.nanmean(mod.allbetas[k], axis=1)
                crit = np.array([crit, -crit]).T
                criti = np.nanmean(mod.allitcpt[k])
                criti = np.array([criti, -criti]).T
            else:
                crit = np.nanmean(mod.allbetas[k], axis=2)
                criti = np.nanmean(mod.allitcpt[k], axis=1)

            k_collect[k] = np.append(k_collect[k], crit, axis=1)

    # step 3: force k threshold
    avgclus = [None] * n_threshs
    passcomp = np.full([n_threshs, n_k], 0)
    for kthresh in range(n_threshs):  # (n_k):
        avgclus[kthresh] = [None] * n_k
        for k in range(n_k):
            # if k >0:#>= kthresh:
            try:
                if link == 0:
                    clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='ward').fit(k_collect[k].T)
                elif link == 1:
                    clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='complete').fit(k_collect[k].T)
                elif link == 2:
                    clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='average').fit(k_collect[k].T)
                else:
                    print('please select linkage function')
                avgbs = np.full([k_collect[k].shape[0], kthresh + 2], np.nan)
                for c in range(kthresh + 2):
                    avgbs[:, c] = np.nanmean(k_collect[k][:, np.where(clustering.labels_ == c)[0]], axis=1)
                avgclus[kthresh][k] = (avgbs)
            except:
                passcomp[kthresh, k] = 1

    errormat = np.full([n_threshs, n_k], np.nan)
    # step 4: calculate match between each solution
    averageerror = np.zeros(n_k)
    for kthresh in range(n_threshs):
        for k in range(n_k):
            if passcomp[kthresh, k] == 0:
                bestmatch = []
                for kmatch in range(len(avgclus[kthresh])):
                    if passcomp[kthresh, kmatch] == 0:
                        if k != kmatch:
                            allmses = np.full([kthresh + 2, kthresh + 2], np.nan)
                            for c1 in range(kthresh + 2):
                                for c2 in range(kthresh + 2):
                                    a = avgclus[kthresh][k][:, c1]
                                    b = avgclus[kthresh][kmatch][:, c2]
                                    allmses[c1, c2] = vector_mse(a, b)
                            for c in range(kthresh + 2):
                                a = np.where(allmses == np.nanmin(allmses))
                                bestmatch.append(np.nanmin(allmses))
                                allmses[a[0][0], :] = np.nan
                                allmses[:, a[1][0]] = np.nan
                errormat[kthresh, k] = np.nanmean(bestmatch)
            # averageerror[kthresh] = np.nanmean(np.array(bestmatch))
    return errormat


def agglomerrorwrap(input_filediri, modstri, n_cv_folds, sets, setsize, n_k):
    allerror = np.full([12, 4, 8, 3], np.nan)
    for s in range(12):
        print(sets[s])
        for mf in range(4):
            for link in range(3):
                averageerror = doagglomchks(s, link, input_filediri, modstri, mf, n_cv_folds, sets, setsize, n_k)
                allerror[s, mf, :, link] = averageerror
    with open((input_filediri + 'allerror.pkl'), 'wb') as f:
        pickle.dump([allerror], f)


# stick everything into one function and do it for all sets
def doagglom_kthresh(s, link, input_filediri, modstri, mf, n_cv_folds, sets, setsize, n_k, kthresh):
    # step 1: collect all vectors across subfolds for each k
    k_collect = [None] * n_k

    for k in range(n_k):

        k_collect[k] = np.empty((setsize[s], 0), int)
        for sf in range(n_cv_folds):
            fold = (mf * n_cv_folds) + sf
            filestr = (input_filediri + sets[s] + modstri + str(fold))
            with open(filestr, "rb") as f:
                mod = pickle.load(f)

            if k == 0:
                crit = np.nanmean(mod.allbetas[k], axis=1)
                crit = np.array([crit, -crit]).T
                criti = np.nanmean(mod.allitcpt[k])
                criti = np.array([criti, -criti]).T
            else:
                crit = np.nanmean(mod.allbetas[k], axis=2)
                criti = np.nanmean(mod.allitcpt[k], axis=1)

            k_collect[k] = np.append(k_collect[k], crit, axis=1)

    # step 3: force k threshold
    avgclus = []
    for k in range(n_k):
        if k >= kthresh:
            if link == 0:
                clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='ward').fit(k_collect[k].T)
            elif link == 1:
                clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='complete').fit(k_collect[k].T)
            elif link == 2:
                clustering = AgglomerativeClustering(n_clusters=kthresh + 2, linkage='average').fit(k_collect[k].T)
            else:
                print('please select linkage function')
            avgbs = np.full([k_collect[k].shape[0], kthresh + 2], np.nan)
            for c in range(kthresh + 2):
                avgbs[:, c] = np.nanmean(k_collect[k][:, np.where(clustering.labels_ == c)[0]], axis=1)
            avgclus.append(avgbs)

    # step 4: calculate match between each solution
    # treat the smallest k as the index
    newbs = np.full([avgclus[0].shape[0], kthresh + 2, len(avgclus)], np.nan)
    for k in range(kthresh + 2):
        newbs[:, k, 0] = avgclus[0][:, k]

    for mf1 in range(len(avgclus)):
        if mf1 != 0:
            allmses = np.full([kthresh + 2, kthresh + 2], np.nan)
            for c1 in range(kthresh + 2):
                for c2 in range(kthresh + 2):
                    a = avgclus[mf1][:, c1]
                    b = avgclus[0][:, c2]
                    allmses[c1, c2] = vector_mse(a, b)
            for c in range(kthresh + 2):
                a = np.where(allmses == np.nanmin(allmses))
                allmses[a[0][0], :] = np.nan
                allmses[:, a[1][0]] = np.nan
                newbs[:, a[1][0], mf1] = avgclus[mf1][:, a[0][0]]

    # fig=plt.figure(figsize=[15,6])
    # for c in range(newbs.shape[1]):
    #    plt.subplot(2,newbs.shape[1],c+1); plt.plot(newbs[:,c,:]);
    #    plt.subplot(2,newbs.shape[1],c+newbs.shape[1]+1); plt.imshow(np.corrcoef(newbs[:,c,:].T)); plt.colorbar();
    # plt.show()

    return avgclus, newbs, k_collect


def cross_sf_similarity_chk(input_filediri,modstri,sets, setsize,n_k=8, n_cv_folds=4):
    averageerror = np.full([len(sets),n_cv_folds,n_k],np.nan)

    for s in range(len(sets)):
        print(sets[s])
        for mf in range(n_cv_folds):

            # collect all betas
            betacollect = [None]*n_k
            for k in range(n_k):
                betacollect[k]=np.full([setsize[s],k+2,n_cv_folds],np.nan)
            for sf in range(n_cv_folds):
                filestr = (input_filediri + sets[s] + modstri + str((mf * n_cv_folds + sf)))
                with open(filestr, "rb") as f:
                    mod = pickle.load(f)
                for k in range(n_k):
                    if k==0:
                        betacollect[k][:,0,sf] = np.nanmean(mod.allbetas[k], axis=1)
                        betacollect[k][:,1,sf] = -np.nanmean(mod.allbetas[k], axis=1)
                    else:
                        betacollect[k][:,:,sf]=np.nanmean(mod.allbetas[k],axis=2)


            for k in range(n_k):
                 # step 4: calculate match between each solution
                bestmatch=[]
                for sf1 in range( n_cv_folds):
                    for sf2 in range( n_cv_folds):
                        if sf1 != sf2:
                            allmses = np.full([k + 2, k + 2], np.nan)
                            for c1 in range(k + 2):
                                for c2 in range(k + 2):
                                    a = betacollect[k][:,c1,sf1]
                                    b = betacollect[k][:,c2,sf2]
                                    if len(np.where(np.isnan(a))[0])==0 and len(np.where(np.isnan(b))[0])==0:
                                        allmses[c1, c2] = vector_mse(a, b)

                            while len(np.where(np.isfinite(allmses))[0])>0:
                                a = np.where(allmses == np.nanmin(allmses))
                                bestmatch.append(np.nanmin(allmses))
                                allmses[a[0][0], :] = np.nan
                                allmses[:, a[1][0]] = np.nan

                averageerror[s,mf,k] = np.nanmean(np.array(bestmatch))

    with open((input_filediri + 'averageerror_samek.pkl'), 'wb') as f:
        pickle.dump(averageerror, f)


def samekagglom_error_mf(input_filedir, sets, setsize, n_k, n_cv_folds):
    error=np.full([len(sets),n_k],np.nan)
    for s in range(len(sets)):
        for k in range(n_k):
            # collect all betas
            betacollect = np.full([setsize[s],k+2,n_cv_folds],np.nan)
            for mf in range(n_cv_folds):
                with open((input_filedir + sets[s] + '_aggr_betas_k' + str(k) + '_mf' + str(mf) + '.pkl'), 'rb') as f:
                    [X, clustering, assig, allbetas, tmp_testproba, argmaxYtest, argmaxYtest2] = pickle.load(f)
                betacollect[:, :, mf] = allbetas

            bestmatch = []
            for mf1 in range(n_cv_folds):
                for mf2 in range(n_cv_folds):
                    if mf1 != mf2:
                        allmses = np.full([k + 2, k + 2], np.nan)
                        for c1 in range(k + 2):
                            for c2 in range(k + 2):
                                a = betacollect[:, c1, mf1]
                                b = betacollect[:, c2, mf2]
                                allmses[c1, c2] = vector_mse(a, b)
                        for c in range(k + 2):
                            a = np.where(allmses == np.nanmin(allmses))
                            bestmatch.append(np.nanmin(allmses))
                            allmses[a[0][0], :] = np.nan
                            allmses[:, a[1][0]] = np.nan
            error[s, k] = np.nanmean(np.array(bestmatch))
    with open((input_filedir + 'mferror_samek.pkl'), 'wb') as f:
        pickle.dump(error, f)


def transformlabels(Y, nclus):
    whoclus = np.full([4, nclus], np.nan)
    c = np.full([3, nclus, nclus], np.nan)
    axis1who = [0, 1, 2];
    axis2who = [1, 2, 3]
    c[0, :, :] = contingency_matrix(Y[:, 0], Y[:, 1])[:nclus, :nclus]
    c[1, :, :] = contingency_matrix(Y[:, 1], Y[:, 2])[:nclus, :nclus]
    c[2, :, :] = contingency_matrix(Y[:, 2], Y[:, 3])[:nclus, :nclus]

    for l in range(3):
        mf1 = axis1who[l];
        mf2 = axis2who[l]
        for k in range(nclus):
            idx = np.where(c[l, :, :] == np.max(c[l, :, :]))
            if l == 0:
                whoclus[mf1, idx[0]] = k
                whoclus[mf2, idx[1]] = k
            else:
                whoclus[mf2, idx[1]] = whoclus[mf1, idx[0]]
            c[l, idx[0][0], :] = 0;
            c[l, :, idx[1][0]] = 0

    Ybar = np.full([Y.shape[0], Y.shape[1]], np.nan)
    for mf in range(4):
        for k in range(nclus):
            Ybar[np.where(Y[:, mf] == k)[0], mf] = whoclus[mf, k]
    return Ybar


# based on these finding we will look at k=4 and k=7 for all solutions
pref = ['normative_correction/FEB_', 'MDD__', 'MDD_spectral_', 'IXI3_', 'IXI2_', 'IXI2_spectral_', 'ALLALL3_']


def get_labels_rand(d, k, n):
    sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s",
            "Tct_Scs_tc_sc_s"]
    dattype = ['MDD_GMM', 'MDD_GMM_null', 'MDD_spectral', 'IXI3_GMM', 'IXI_GMM_null', 'IXI_spectral', 'ALL_GMM']
    modstr = ['_mod_ctrl_', '_mod_null_', '_mod_', '_mod_', '_mod_null_', '_mod_', '_mod_']
    modpref = ['FEB_', 'MDD__', 'MDD_spectral_', 'IXI3_', 'IXI2_', 'IXI2_spectral_', 'ALLALL3_']
    pref = ['FEB_', 'MDD__', 'MDD_spectral_', 'IXI3_', 'IXI2_', 'IXI2_spectral_', 'ALLALL3_']

    with open('/Volumes/ELEMENTS/clustering_pilot/clustering_output/' + dattype[d] + '/summaries/' + pref[
        d] + 'agglom_moremet.pkl', 'rb') as f:
        [averageerror, n_vecs, source_vecs, ed_btw_vecs, avgclus] = pickle.load(f)

    ALL_testlabels_k4 = np.full([n, 12], np.nan)
    for s in range(12):
        with open(('/Volumes/ELEMENTS/clustering_pilot/clustering_output/' + dattype[d] + '/mod/' + modpref[d] + sets[s] +
                   modstr[d] + str(0)), 'rb') as f:
            mod = pickle.load(f)
        X = mod.data
        Y = np.full([X.shape[0], 4], np.nan)
        for mf in range(4):
            betas = avgclus[s][mf][k - 2][k - 2]
            Y[:, mf] = predictargmax(X, np.concatenate([np.ones(shape=[1, k]), betas], axis=0))
        Ybar = transformlabels2(Y, k)
        for mf in range(4):
            tests = np.where(np.isnan(mod.cv_assignment[:, mf]))[0]
            ALL_testlabels_k4[tests, s] = Ybar[tests, mf]

    rand_ALL_k4 = np.full([12, 12], np.nan)
    for s1 in range(12):
        for s2 in range(12):
            rand_ALL_k4[s1, s2] = rand_score_withnans(ALL_testlabels_k4[:, s1], ALL_testlabels_k4[:, s2])
    return ALL_testlabels_k4, rand_ALL_k4


def loadset(d, s):
    sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s",
            "Tct_Scs_tc_sc_s"]
    dattype = ['MDD_GMM', 'MDD_GMM_null', 'MDD_spectral', 'IXI3_GMM', 'IXI_GMM_null', 'IXI_spectral', 'ALL_GMM']
    modstr = ['_mod_ctrl_', '_mod_null_', '_mod_', '_mod_', '_mod_null_', '_mod_', '_mod_']
    pref = ['FEB_', 'MDD__', 'MDD_spectral_', 'IXI3_', 'IXI2_', 'IXI2_spectral_', 'ALLALL3_']

    with open(('/Volumes/ELEMENTS/clustering_pilot/clustering_output/' + dattype[d] + '/mod/' + pref[d] + sets[s] +
               modstr[d] + str(0)), 'rb') as f:
        mod = pickle.load(f)
    return mod.data, mod.cv_assignment


def transformlabels2(Y, nclus):
    whoclus = np.full([4, nclus], np.nan)
    c = np.full([4, nclus, nclus], np.nan)
    axis1who = [0, 1, 2, 3];
    axis2who = [1, 2, 3, 0]
    c[0, :, :] = contingency_matrix(Y[:, 0], Y[:, 1])[:nclus, :nclus]
    c[1, :, :] = contingency_matrix(Y[:, 1], Y[:, 2])[:nclus, :nclus]
    c[2, :, :] = contingency_matrix(Y[:, 2], Y[:, 3])[:nclus, :nclus]
    c[3, :, :] = contingency_matrix(Y[:, 3], Y[:, 0])[:nclus, :nclus]
    # print(c)

    for k in range(nclus):
        idx = np.where(c == np.max(c))
        # print(np.max(c))
        mf1 = axis1who[idx[0][0]]
        mf2 = axis2who[idx[0][0]]
        mf1clus = idx[1][0]
        mf2clus = idx[2][0]

        if np.isnan(whoclus[mf1, mf1clus]) and np.isnan(whoclus[mf2, mf2clus]):
            whoclus[mf1, mf1clus] = k
            whoclus[mf2, mf2clus] = k

            if idx[0][0] == 0:  # we have 0 and 1

                # mf2==1: # fill in 2
                crit = np.array(c[1, np.where(whoclus[1, :] == k)[0], :])[0]
                if np.max(crit) > 10:
                    whoclus[2, np.where(crit == np.max(crit))[0]] = k
                # mf1==0: # fill in 3
                crit = np.array(c[3, :, np.where(whoclus[0, :] == k)[0]])[0]
                if np.max(crit) > 10:
                    whoclus[3, np.where(crit == np.max(crit))[0]] = k

            elif idx[0][0] == 1:  # we have 1 and 2

                # mf1==1: # fill in 0
                crit = np.array(c[0, :, np.where(whoclus[1, :] == k)[0]])[0]
                if np.max(crit) > 10:
                    whoclus[0, np.where(crit == np.max(crit))[0]] = k
                # mf2==2: # fill in 3
                crit = np.array(c[2, np.where(whoclus[2, :] == k)[0], :])[0]
                if np.max(crit) > 10:
                    whoclus[3, np.where(crit == np.max(crit))[0]] = k

            elif idx[0][0] == 2:  # we have 2 and 3

                # mf1==2: # fill in 1
                crit = np.array(c[1, :, np.where(whoclus[2, :] == k)[0]])[0]
                if np.max(crit) > 10:
                    whoclus[1, np.where(crit == np.max(crit))[0]] = k
                # mf2==3: # fill in 0
                crit = np.array(c[3, np.where(whoclus[3, :] == k)[0], :])[0]
                if np.max(crit) > 10:
                    whoclus[0, np.where(crit == np.max(crit))[0]] = k

            elif idx[0][0] == 2:  # we have 3 and 0

                # mf1==3: # fill in 2
                crit = np.array(c[2, :, np.where(whoclus[3, :] == k)[0]])[0]
                if np.max(crit) > 10:
                    whoclus[2, np.where(crit == np.max(crit))[0]] = k
                # mf2==0: # fill in 1
                crit = np.array(c[0, np.where(whoclus[0, :] == k)[0], :])[0]
                if np.max(crit) > 10:
                    whoclus[1, np.where(crit == np.max(crit))[0]] = k

            # print(whoclus)
            for l in range(4):
                c[l, np.where(whoclus[axis1who[l], :] == k)[0], :] = 0
                c[l, :, np.where(whoclus[axis2who[l], :] == k)[0]] = 0

        else:
            print('stop')
            # print(whoclus)
            # print(c)
    print(whoclus)
    Ybar = np.full([Y.shape[0], Y.shape[1]], np.nan)
    for mf in range(4):
        for k in range(nclus):
            Ybar[np.where(Y[:, mf] == k)[0], mf] = whoclus[mf, k]
    return Ybar