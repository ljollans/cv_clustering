import pickle
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')
from cv_clustering.mainfoldaggr import agglom
import os
from sklearn.metrics import silhouette_samples, silhouette_score

# outcome metrics I want:

# Level 1: LOOCV clustering
    # silhouette score
    # proportion of ambiguous clustering
    # cluster sizes

# Level 2: aggregated LOOCV solutions in each subfold
    # silhouette score after cluster ensemble
    # silhouette score after new classification model
    # micro f1 for new classification model
    # macro f1 for new classification model
    # test set classification probability
    # cluster sizes

# Level 3: aggregated subfold solutions in each mainfold
    # silhouette score
    # test set classification probability

# Level 4: commonality across mainfold solutions


# analysis run info:
input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'
modstr = '_mod_ctrl_'
#input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/null/MDDnull/MDD__'
#modstr = '_mod_null_'
#input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/IXI2/null2/IXI2_'
#modstr = '_mod_null_'
#input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/IXI2/act/IXI2_'
#modstr = '_mod_'

sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s", "Tct_Scs_tc_sc_s"]
n_cv_folds = 4
n = 398
#n = 544
n_k = 8

do_level_1 = 0
do_level_2 = 0
do_level_3 = 1

pac_lvl1_done = 1
reclass_lvl2_done = 1
agglom_lvl3_done = 0
moveon=0

##################
#    LEVEL 1     #
##################

if do_level_1==1:
    # calculate pac
    def calcpac(filestr):
        with open(filestr, "rb") as f:
            mod = pickle.load(f)
        if hasattr(mod,'pac'):
            pass
        else:
            mod.calc_consensus_matrix()
            mod.get_pac()
            with open(filestr, "wb") as f:
                pickle.dump(mod,f)

    if pac_lvl1_done==0:
        for s in range(len(sets)):
            print(sets[s])
            Parallel(n_jobs=8)(delayed(calcpac)((input_filedir + sets[s] + modstr + str(fold))) for fold in range(16))

    clussizeedges=np.arange(0,n,10)
    silhouette_lvl1 = np.full([len(sets),n_cv_folds,n_cv_folds,n,n_k],np.nan)
    pac_lvl1 = np.full([len(sets),n_cv_folds,n_cv_folds,n_k],np.nan)
    clussize_lvl1 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k,len(clussizeedges)-1], np.nan)
    for s in range(len(sets)):
        for mf in range(n_cv_folds):
            for sf in range(n_cv_folds):
                fold=(mf*n_cv_folds)+sf
                filestr=(input_filedir + sets[s] + modstr + str(fold))
                with open(filestr, "rb") as f:
                    mod = pickle.load(f)
                silhouette_lvl1[s,mf,sf,:,:]=mod.sil
                pac_lvl1[s, mf, sf, :] = mod.pac
                for k in range(n_k):
                    tmp=np.full([len(mod.train_index_sub),k+2],np.nan)
                    for ll in range(len(mod.train_index_sub)):
                        for k2 in range(k+2):
                            tmp[ll,k2] = len(np.where(mod.all_clus_labels[mod.train_index_sub[ll],k,:]==k2)[0])
                    h,b=np.histogram(tmp.flatten(), clussizeedges)
                    clussize_lvl1[s,mf,sf,k,:]=h

    with open(input_filedir + modstr + 'sil_pac_lvl1.pkl', 'wb') as f:
        pickle.dump([silhouette_lvl1,pac_lvl1, clussize_lvl1],f)



##################
#    LEVEL 2     #
##################

if do_level_2==1:

    def calcreclass(filestr):
        with open(filestr, "rb") as f:
            mod = pickle.load(f)
        if hasattr(mod,'testset_prob'):
            pass
        else:
            mod.cluster_ensembles_new_classification()
            mod.sf_class_probas()
            with open(filestr, "wb") as f:
                pickle.dump(mod,f)

    if reclass_lvl2_done==0:
        for s in range(len(sets)):
            print(sets[s])
            Parallel(n_jobs=8)(delayed(calcreclass)((input_filedir + sets[s] + modstr + str(fold))) for fold in range(16))

    clussizeedges = np.arange(0, n, 10)
    silhouette1_lvl2 = np.full([len(sets),n_cv_folds,n_cv_folds, n_k],np.nan)
    silhouette2_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan)
    microf1_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan)
    macrof1_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan)
    testproba_lvl2 = np.full([len(sets),n_cv_folds,n_cv_folds, n_k],np.nan)

    clussize_CE_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k, len(clussizeedges) - 1], np.nan)
    clussize_test_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k, len(clussizeedges) - 1], np.nan)
    for s in range(len(sets)):
        for mf in range(n_cv_folds):
            for sf in range(n_cv_folds):
                fold=(mf*n_cv_folds)+sf
                filestr=(input_filedir + sets[s] + modstr + str(fold))
                with open(filestr, "rb") as f:
                    mod = pickle.load(f)
                silhouette1_lvl2[s,mf,sf,:]=mod.silhouette_cluster_ensembles
                silhouette2_lvl2[s, mf, sf, :] = mod.silhouette2_lvl2
                microf1_lvl2[s, mf, sf, :] = mod.micro_f1
                macrof1_lvl2[s, mf, sf, :] = mod.macro_f1
                testproba_lvl2[s, mf, sf, :] = np.nanmean(mod.testset_prob,axis=0)
                for k in range(n_k):
                    tmp=np.full([k+2,2],np.nan)
                    for k2 in range(k+2):
                        tmp[k2, 0] = len(np.where(mod.cluster_ensembles_labels[:,k]==k2)[0])
                        tmp[k2, 1] = len(np.where(mod.testlabels[:, k] == k2)[0])
                    clussize_CE_lvl2[s, mf, sf, k,:],b=np.histogram(tmp[:,0], clussizeedges)
                    clussize_test_lvl2[s, mf, sf, k,:],b=np.histogram(tmp[:,1], clussizeedges)


    with open(input_filedir + modstr + 'sil_f1_prob_lvl2.pkl', 'wb') as f:
        pickle.dump([silhouette1_lvl2,silhouette2_lvl2,microf1_lvl2,macrof1_lvl2,testproba_lvl2,clussize_CE_lvl2,clussize_test_lvl2],f)



##################
#    LEVEL 3     #
##################

if do_level_3 == 1:
    if agglom_lvl3_done == 0:
        agglom(input_filedir, modstr, sets, n_k, n_cv_folds)

    if moveon==1:

        silhouette_dot_lvl3 = np.full([len(sets), n_cv_folds, n_k], np.nan)
        silhouette_prob_lvl3 = np.full([len(sets), n_cv_folds, n_k], np.nan)
        testproba_best_lvl3 = np.full([len(sets), n_cv_folds, n_k], np.nan)
        testproba_best_perclus_lvl3 = np.full([len(sets), n_cv_folds, n_k, n_k+2], np.nan)
        clussize_dot_lvl3 = np.full([len(sets), n_cv_folds, n_k, n_k+2], np.nan)
        clussize_prob_lvl3 = np.full([len(sets), n_cv_folds, n_k, n_k+2], np.nan)

        for s in range(12):
            filestr = (input_filedir + sets[s] + modstr + str(0))
            with open(filestr, "rb") as f:
                mod = pickle.load(f)
            for mf in range(n_cv_folds):
                maintrain = np.where(np.isfinite(mod.cv_assignment[:, mf]))[0]
                maintest = np.where(np.isnan(mod.cv_assignment[:, mf]))[0]
                for k in range(n_k):
                    with open((input_filedir + sets[s] + '_aggr_betas_k' + str(k) + '_mf' + str(mf) + '.pkl'),'rb') as f:
                        [X, clustering, assig, allbetas, tmp_testproba, argmaxYdot, argmaxYprob] = pickle.load(f)
                    silhouette_dot_lvl3[s,mf,k] = silhouette_score(mod.data[maintest], argmaxYdot)
                    silhouette_prob_lvl3[s, mf, k] = silhouette_score(mod.data[maintest], argmaxYprob)
                    topprob = np.array([np.max(tmp_testproba[i,:]) for i in range(tmp_testproba.shape[0])])
                    testproba_best_lvl3[s,mf,k] = np.nanmean(topprob)
                    for c in range(k+1):
                        testproba_best_perclus_lvl3[s,mf,k,c]=np.nanmean(topprob[np.where(argmaxYprob==c)[0]])
                        clussize_dot_lvl3[s,mf,k,c]=len(np.where(argmaxYdot==c)[0])
                        clussize_prob_lvl3[s,mf,k,c]=len(np.where(argmaxYprob==c)[0])



