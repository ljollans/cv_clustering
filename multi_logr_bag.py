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
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

# function for carrying out the regression
def log_in_CV(x2use,y2use, nclus):
    # we'll be making a classifier for each group separately
    allbetas=np.zeros(shape=[x2use.shape[1],nclus]) # pre-allocate the array because we'll be filling it step by step
    u=set(y2use) # the unique values for y, i.e. the groups we have
    ctr=-1 # we're starting a counter because the groups might be [2,3,4] but we want to populate the array as [0,1,2]
    for n in u: # loop over the groups
        ctr+=1
        y=np.zeros(shape=[y2use.shape[0]])
        y[np.where(y2use==n)]=1
        # now there are just 2 groups: those in my target group and those outside my target group
        clf = LogisticRegression(random_state=0, solver='lbfgs')
        try:
            res=clf.fit(x2use,y)
#        try:
            allbetas[:,n.astype(int)]=clf.coef_ # n instead of ctr because i want to maintain the correct order even if a class is missing in this subset of the data
#        except:
#            print(Yboot.shape)
#            print(np.sum(Yboot))
        except:
            print('only one class for '+ str(nclus))
    return allbetas

# define function to take subsets of the sample for bootstrap aggregation
from random import randrange
def subsample(dataset, ratio=1.0):
    totalN=dataset.shape[0]
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
def bag_log(x2use,y2use,nboot, nclus):
    allbetas=np.zeros(shape=[x2use.shape[1],nclus,nboot]) # pre-allocate
    for i in range(nboot):
        # get and select the bootstrap sample
        bootidc = subsample(x2use, 0.666) 
        Xboot = x2use[bootidc,:]
        Yboot = y2use[bootidc]
        # run it through the regression and save regression weights
        allbetas[:,:,i]=log_in_CV(Xboot,Yboot, nclus)
    # average regression weight from all bootstrap runs
    betas=np.nanmean(allbetas,axis=2)
    return betas # and spit them out again at the end

# make a function to get the AUC and F1 values
# get the  AUC and F1 values to check how good the model actually is
def getmetrics(x2use, truth,betas2use):

    pred=np.squeeze(x2use.dot(betas2use))
    newpred=np.zeros(shape=[pred.shape[0]])
    

    aucs=np.nan
    f1s=np.nan
    
    try:
        fpr,tpr,thresholds=roc_curve(truth,pred)
        point=np.argmax(tpr-fpr)

#        newpred=np.zeros(shape=[pred.shape[0]])
        newpred[np.where(pred>=thresholds[point])]=1

#        aucs=np.nan
#        f1s=np.nan
        
        #try:
        aucs=roc_auc_score(truth,newpred)
        f1s=f1_score(truth,newpred)
    except:
        print('couldnt calculate ROC') 
         
#except:
            #print('max:' + str(np.nanmax(newpred)) +', min:' + str(np.nanmin(newpred)) + ',  nans:' + str(np.isnan(newpred).sum()))
    #else:
        #print('WARNING! there were ' + str(len(set(newpred))) + ' groups predicted with truth having ' + str(len(set(truth))) + ' groups.')
    
    return aucs, f1s, newpred, pred

from sklearn.model_selection import KFold
def multi_logr_bagr(nboot,XYtrain, nclus, ncv, output):
    Y=XYtrain[:,0]
    X=XYtrain[:,1:]
    # define cross-validation
    nfolds=ncv
    kf = KFold(n_splits=nfolds)

    #pre-allocate arrays
    overallpred=np.full([len(Y),nclus],np.nan)
    overallpred0=np.full([len(Y),nclus],np.nan)
    overallpred1=np.full([len(Y),nclus],np.nan)
    aucs_partial=np.full([nfolds,nclus],np.nan)
    f1s_partial=np.full([nfolds,nclus],np.nan)
    betas=np.full([X.shape[1],nclus,ncv], np.nan) 
    problemrec=np.zeros(shape=[nfolds,nclus])

    fold=-1
    for train_index, test_index in kf.split(X):
        fold+=1
        #print('fold '+ str(fold))
        x2use=X[train_index,:]
        y2use=Y[train_index]
        nboot=10
        # make prediction model based on training set
        if nclus>1:
            betas[:,:,fold]=bag_log(x2use,y2use,nboot, nclus) 
        else:
            betas[:,:,fold]=log_in_CV(x2use,y2use, nclus)
        # predict test set groups
        x2use=X[test_index,:]
        y2use=Y[test_index]
        u=set(y2use)
        for n in u:
            n_y2use=np.zeros(shape=[len(y2use)])
            n_y2use[np.where(y2use==n)[0]]=1
            [aucs_partial[fold,n.astype(int)],f1s_partial[fold,n.astype(int)], newpred, pred]=getmetrics(x2use, n_y2use,betas[:,n.astype(int),fold])
            overallpred0[test_index,n.astype(int)]=pred
            overallpred[test_index,n.astype(int)]=newpred

    #pre-allocate arrays
    aucs=np.full([len(set(Y))], np.nan)
    betas2use=np.nanmean(betas,axis=2)
    f1s=np.full([len(set(Y))],np.nan)
    # considering only the test sets constituting the whole sample, get model fit metrics
    for n in range(nclus):
        truth=np.zeros(shape=[len(Y)])
        truth[np.where(Y==n)[0]]=1

        if len(set(overallpred[:,n]))>1:
            try:
                aucs[n]=roc_auc_score(truth,overallpred[:,n])
                f1s[n]=f1_score(truth,overallpred[:,n])
            except:
                #print('problem with prediction - max:' + str(np.nanmax(overallpred[:,n])) +', min:' + str(np.nanmin(overallpred[:,n])) + ',  nans:' + str(np.isnan(overallpred[:,n]).sum()) + ' of N=' + str(len(truth)))
                problemrec[fold,n]=1
        else:
            #print('WARNING! there were ' + str(len(set(overallpred[:,n]))) + ' groups predicted with truth having ' + str(len(set(truth))) + ' groups.')
            problemrec[fold,n]=2
        if output==1:
                print('group '+ str(n) + ': AUC='+ str(aucs) + ', F1 score='+ str(f1s))

    groupclass=np.zeros(shape=[X.shape[0]])
    for n in range(len(set(Y))):
        groupclass[np.where(np.max(overallpred1-np.expand_dims(overallpred1[:,n],axis=1),axis=1)==0)]=n
    # plot overall prediction
    correctclass=np.zeros(shape=[X.shape[0]])
    correctclass[np.where((Y-groupclass)==0)[0]]=1
    if output==1:
        fig=plt.figure(figsize=[20,4])
        fig.add_subplot(1,3,1); plt.scatter(X[:,0],X[:,1],c=Y); plt.title('Actual groups')
        fig.add_subplot(1,3,2); plt.scatter(X[:,0],X[:,1],c=groupclass); plt.title('Predicted groups')
        fig.add_subplot(1,3,3); plt.scatter(X[:,0],X[:,1],c=correctclass); plt.title('Correct classifications')
        plt.show()

    return aucs, f1s, betas2use, overallpred0, overallpred, aucs_partial, f1s_partial, betas, groupclass, correctclass, problemrec
