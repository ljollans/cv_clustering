#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:14:07 2020

@author: lee_jollans
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import os

from multi_logr_bag import multi_logr_bagr


def clusmets(A):
    X2=A[0]
    nclus=A[1]
    # cluster
    gmm = GaussianMixture(n_components=nclus+2, covariance_type=A[2], n_init=100,random_state=10).fit(X2) 
                          # here we could theoretically tune for covariance type and clustering method
    # get cluster membership
    cluslabels=gmm.predict(X2)
    # get cluster fit metrics
    BIC=gmm.bic(X2)
    SIL=metrics.silhouette_score(X2,cluslabels)
    CAL=metrics.calinski_harabasz_score(X2,cluslabels)
    # prep data for multi-class classification
    x=X2[np.where(cluslabels!=-1)[0],:] # not sure why i added this but there shouldn't be anyone with no assignment
    y=cluslabels[np.where(cluslabels!=-1)[0]]
    XYtrain=np.append(np.expand_dims(y,1),x,axis=1)
    # classify
    [tmpauc, tmpf1, betas2use, overallpred0, overallpred, aucs_partial, f1s_partial, tmpbetas, groupclass, correctclass, problemrec]=multi_logr_bagr(10,XYtrain, nclus+2, 4, 0)
    #print(problemrec)
    AUC=np.mean(tmpauc)
    F1=np.mean(tmpf1)
    BETAS=np.nanmean(tmpbetas,axis=2).T

    return cluslabels, BIC, SIL, CAL, AUC, F1, BETAS
