#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
os.chdir('/u/ljollans/ML_in_python/export_251019/jan2020')
import clusmets
from clusmets import clusmets


def clusmetwrapper(X, CVassig,mainfold,subfold,covariance2use,n_ks):

    n_samples=X.shape[0]
    n_features=X.shape[1]

    allcluslabels=np.full([n_samples,n_ks,n_samples],np.nan)
    BIC = np.full([n_samples, n_ks], np.nan)
    SIL = np.full([n_samples, n_ks], np.nan)
    CAL = np.full([n_samples, n_ks], np.nan)
    AUC = np.full([n_samples, n_ks], np.nan)
    F1 = np.full([n_samples, n_ks], np.nan)
    BETAS=np.full([n_samples, n_ks,n_ks+2,n_features],np.nan)

    #train_index_main=np.where(CVassig[:,mainfold]>=0)[0]
    train_index_sub=np.where((CVassig[:,mainfold]!=subfold) & (~np.isnan(CVassig[:,mainfold])))[0]
    X2=X[train_index_sub,:]
    
    #looco=0
    for looco in range(len(train_index_sub)):
        X3=np.delete(X2,looco,axis=0)
        train_index_sub_copy=np.delete(train_index_sub,looco,axis=0)
        
        #return cluslabels, BIC, SIL, CAL, AUC, F1, BETAS
        for nclus in range(n_ks):
            [tmpcluslabels, tmpBIC, tmpSIL, tmpCAL, tmpAUC, tmpF1, tmpBETAS]=clusmets([X3, nclus,covariance2use])
            allcluslabels[train_index_sub[looco],nclus,train_index_sub_copy]=tmpcluslabels
            BIC[train_index_sub[looco],nclus]=tmpBIC
            SIL[train_index_sub[looco],nclus]=tmpSIL
            CAL[train_index_sub[looco],nclus]=tmpCAL
            AUC[train_index_sub[looco],nclus]=tmpAUC
            F1[train_index_sub[looco],nclus]=tmpF1
            BETAS[train_index_sub[looco],nclus,:nclus+2,:]=tmpBETAS

    return allcluslabels,BIC,SIL,CAL,AUC,F1,BETAS
