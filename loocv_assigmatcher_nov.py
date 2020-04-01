# load LOOCV cluster assignments and get a consensus cluster naming
import sys
import os
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv

from utils import select_trainset, n_clus_retrieval_chk


def loocv_assigmatcher(all_cluslabels):
    if all_cluslabels.shape[1]!=all_cluslabels.shape[0]:
        sys.exit('all_cluslabels should be an NxN np-array')
    N=all_cluslabels.shape[0]
    met=np.full([N,N],np.nan)
    for ppt1 in range(N):
        for ppt2 in range(N):
            if ppt1!=ppt2:
                a1=all_cluslabels[ppt1,:]; a2=all_cluslabels[ppt2,:]
                a2=np.delete(a2,np.where(np.isnan(a1))[0]); a1=np.delete(a1,np.where(np.isnan(a1))[0]); 
                a1=np.delete(a1,np.where(np.isnan(a2))[0]); a2=np.delete(a2,np.where(np.isnan(a2))[0]); 
                if len(a1)>1:
                    C=contingency_matrix(a1,a2)
                    correcta=np.full([C.shape[0]],np.nan); incorrecta=np.full([C.shape[0]],np.nan)
                    for c in range(C.shape[0]):
                        mmax=np.max(C[c,:])
                        msum=np.sum(C[c,:])
                        correcta[c]=mmax
                        incorrecta[c]=msum-mmax
                    met[ppt1,ppt2]=(np.sum(incorrecta)*100)/len(a2)
    
    return met

def getsubfoldbetas(savedir,s,ctr, nclus,null):
    sets=['Tc','Sc','TSc','Tc_tc','Sc_sc','TSc_tsc','Tct_s','Scs_s','Tct_Scs_s','Tct_tc_s','Scs_sc_s','Tct_Scs_tc_sc_s']
    
    # check how many variables
    if null==1:
        if ctr==1:
            str2add='_null_ctrl_fold'
        else:
            str2add='_null_fold'
    else:
        if ctr==1:
            str2add='_ctrl_fold'
        else:
            str2add='_fold'
    pkl_filename=(savedir + sets[s] + 'BETAS' + str2add + str(0) + '.pkl')
    with open(pkl_filename, 'rb') as file: 
        BETAS=pickle.load(file)
    #allocate array
    print(BETAS.shape)

    allbetas=np.full([4,4,nclus+2,BETAS[0].shape[2]],np.nan)
    for mainfold in range(4):
        for subfold in range(4):

            allcluslabels=np.full([398,8,398],np.nan)
            fold=(4*mainfold)+subfold
           
            pkl_filename = ((savedir + sets[s] + 'allcluslabels' + str2add + str(fold) + '.pkl'))
           
            with open(pkl_filename, 'rb') as file:
                allcluslabels=pickle.load(file)

            all_cluslabels=allcluslabels[:,nclus.astype(int),:]
            met=loocv_assigmatcher(all_cluslabels)
            bestidx=np.where(np.nanmin(met,axis=0)==np.nanmax(np.nanmin(met,axis=0)))[0]
            newidxs=match2idx(all_cluslabels[bestidx[0],:], all_cluslabels, nclus.astype(int))

            tmpbetas=np.full([398,nclus+2,BETAS[0].shape[2]],np.nan)
            fold=(4*mainfold)+subfold
            pkl_filename1=(savedir + sets[s] + 'BETAS' + str2add  + str(fold) + '.pkl')
            pkl_filename2=(savedir + sets[s] + 'allcluslabels' + str2add + str(fold) + '.pkl') 
            with open(pkl_filename1, 'rb') as file: 
                BETAS=pickle.load(file)
            with open(pkl_filename2, 'rb') as file: 
                allcluslabels=pickle.load(file)

            for ppt in range(398):
                tmp1=allcluslabels[ppt,nclus,:] # so the LOOCV run
                tmp2=newidxs[ppt,:]
                m=np.full([nclus+2],np.nan)
                if len(np.where(np.isnan(tmp1))[0])<300:
                    flag=0
                    for c in range(nclus+2):
                        # first check which BETA vector corresponds to which new assignment
                        a2=np.where(tmp2==c)[0]
                        if a2.size==0:
                            flag=1
                        else:
                            if len(set(tmp1[a2]))>1:
                                print(tmp1[a2])
                                sys.exit('clusters in newidxs not perfectly matched to allcluslabels')
                            if tmp1[a2[0]].astype(int) in m:
                                sys.exit('somehow assigned the same value to two clusters')
                                
                            m[c]=tmp1[a2[0]].astype(int)
                            tmpbetas[ppt,c,:]=BETAS[ppt][nclus,m[c].astype(int),:]
                    if flag==1:
                        # missed out a beta because there was no one assigned to it
                        f=np.where(np.isnan(m))[0]
                      #  if len(f)==1:
                        ff=np.setdiff1d(np.arange(nclus+2),m)
                        tmpbetas[ppt,np.where(np.isnan(m))[0],:]=BETAS[ppt][nclus,ff[0],:]
                      #  else:
                      #      sys.exit('more than one flag')
            allbetas[mainfold,subfold,:,:]=np.nanmean(tmpbetas,axis=0)
    return allbetas


def match2idx(idx, tomatch,nclus):
    N=len(idx)
    print(nclus)
    newidxs=np.full([tomatch.shape[1],N],np.nan)
    for p in range(tomatch.shape[1]):
        C=contingency_matrix(idx,tomatch[p,:])
        C=C[:nclus+2,:nclus+2]
        for c in range(nclus+2):
            if np.max(C)>0:
                maxw=np.where(C==np.max(C))
                clusnew=maxw[0][0]
                clusold=maxw[1][0]
                tmp1=np.where(tomatch[p,:]==clusold)[0]
                newidxs[p,tmp1]=clusnew
                C[clusnew,:]=np.zeros(shape=[nclus+2])
                C[:,clusold]=np.zeros(shape=[nclus+2])
    return newidxs


def getSILCALBIC(savedir,ex1, null):
    sets=['Tvc','Svc','TSvc','Tvc_tvc', 'Svc_svc','TSvc_tsvc', 'Tvct_s','Svcs_s','Tvct_Svcs_s','Tvct_tvc_s','Svcs_svc_s','Tvct_Svcs_tvc_svc_s']
    #sets=['Tc','Sc','TSc','Tc_tc','Sc_sc','TSc_tsc','Tct_s','Scs_s','Tct_Scs_s','Tct_tc_s','Scs_sc_s','Tct_Scs_tc_sc_s']
        
# get silhouette scores for all
    
    SIL=np.full([398,8,4,4,2,len(sets)],np.nan)
    CAL=np.full([398,8,4,4,2,len(sets)],np.nan)
    BIC=np.full([398,8,4,4,2,len(sets)],np.nan)
    bestnclus_sf=np.full([4,4,2,len(sets)],np.nan)
    bestnclus_mf=np.full([4,2,len(sets)],np.nan)
    bestSIL=np.full([4,2,len(sets)],np.nan)
    bestCAL=np.full([4,2,len(sets)],np.nan)
    for s in range(len(sets)):
        for ctr in range(2):
            if null==1:
                if ctr==1:
                    str2add='_null_ctrl'
                else:
                    str2add='_null'
            else:
                if ctr==1:
                    str2add='_ctrl'
                else:
                    str2add=''
            if ex1[ctr,s]==16:
                fold=-1
                for mainfold in range(4):
                    for subfold in range(4):
                        fold+=1
                        
                        pkl_filename1 = (savedir + sets[s] + 'SIL' + str2add + '_fold' + str(fold) + '.pkl')
                        pkl_filename2 = (savedir + sets[s] + 'CAL' + str2add + '_fold' + str(fold) + '.pkl')
                        pkl_filename3 = (savedir + sets[s] + 'BIC' + str2add + '_fold' + str(fold) + '.pkl') 
                    
                        with open(pkl_filename1, 'rb') as file: 
                            SIL[:,:,mainfold,subfold,ctr,s]=pickle.load(file)
                        with open(pkl_filename2, 'rb') as file: 
                            CAL[:,:,mainfold,subfold,ctr,s]=pickle.load(file)
                        with open(pkl_filename3, 'rb') as file: 
                            BIC[:,:,mainfold,subfold,ctr,s]=pickle.load(file)
                        tmp=np.nanmean(BIC[:,:,mainfold,subfold,ctr,s],axis=0)
                        bestnclus_sf[mainfold,subfold,ctr,s]=np.where(tmp==np.nanmin(tmp))[0]
                    # get best nclus
                    tmp=np.nanmean(np.nanmean(BIC[:,:,mainfold,:,ctr,s],axis=0),axis=1)
                    bestnclus_mf[mainfold,ctr,s]=np.where(tmp==np.nanmin(tmp))[0]
                    bestSIL[mainfold,ctr,s]=np.nanmean(SIL[:,bestnclus_mf[mainfold,ctr,s].astype(int),mainfold,:,ctr,s])
                    bestCAL[mainfold,ctr,s]=np.nanmean(CAL[:,bestnclus_mf[mainfold,ctr,s].astype(int),mainfold,:,ctr,s])
    return bestSIL,bestCAL,bestnclus_mf, bestnclus_sf, SIL, CAL, BIC


def plot_bic_violin(BIC,mainfold,subfold):
    bic = pd.DataFrame ( {}, columns=[ 'bic', 'k', 'ppt' ] )
    for nclus in range ( BIC.shape[ 1 ] ):
        tmp_df = pd.DataFrame (
            {'bic': np.squeeze ( BIC[ :, nclus, mainfold, subfold, 0, 7 ] ),
             'k': np.ones ( shape=[ 398 ] ) * nclus + 2,
             'ppt': np.arange ( 398 )},
            columns=[ 'bic', 'k', 'ppt' ] )
        bic = bic.append ( tmp_df )
    sns.violinplot ( 'k', 'bic', data=bic )

def k_workup_mainfold(mainfold,set):
    sets = [ 'Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct_s', 'Svcs_s', 'Tvct_Svcs_s', 'Tvct_tvc_s',
             'Svcs_svc_s', 'Tvct_Svcs_tvc_svc_s' ]
    savedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open ( (cv_assignment_dir + "CVassig398.csv"), "r" ) as f:
        reader = csv.reader ( f, delimiter="," )
        cv_assignment = np.array ( list ( reader ) ).astype ( float )

    ex1 = getloopcount ( savedir, 0 )
    [ bestSIL, bestCAL, bestnclus_mf, bestnclus_sf, SIL, CAL, BIC ] = getSILCALBIC ( savedir, ex1, 0 )
    fig = plt.figure ()
    ctr = 0
    for subfold in range ( 4 ):
        ctr += 1;
        plt.subplot ( 4, 3, ctr )
        plot_bic_violin ( BIC, mainfold, subfold );
        plt.title ( 'BIC' )

        # load data
        pkl_filename = (savedir + sets[set] + 'allcluslabels_fold' + str (
            subfold ) + '.pkl')
        with open ( pkl_filename, "rb" ) as file:
            Aorig = pickle.load ( file )
        trainset = select_trainset ( cv_assignment, mainfold, subfold )
        A1 = Aorig[ trainset, :, : ]
        A2 = A1[ :, :, trainset ]

        # cluster "saturation" and PAC score
        cr, pac, plateau_cr = n_clus_retrieval_chk ( A2 )

        ctr += 1;
        plt.subplot ( 4, 3, ctr );
        plt.title ( 'cluster saturation, plateau:' + str ( plateau_cr ) )
        plt.plot ( cr )
        ctr += 1;
        plt.subplot ( 4, 3, ctr );
        plt.title ( 'PAC' )
        plt.plot ( pac )
    plt.show ()

# this is the contingency matrix function used in sklearn

from math import log

import numpy as np
from scipy import sparse as sp

def contingency_matrix(labels_true, labels_pred, eps=None, sparse=False):
    """Build a contingency matrix describing the relationship between labels.
    Parameters
    ----------
    labels_true : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate
    eps : None or float, optional.
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propagation.
        If ``None``, nothing is adjusted.
    sparse : boolean, optional.
        If True, return a sparse CSR continency matrix. If ``eps is not None``,
        and ``sparse is True``, will throw ValueError.
        .. versionadded:: 0.18
    Returns
    -------
    contingency : {array-like, sparse}, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
        Will be a ``scipy.sparse.csr_matrix`` if ``sparse=True``.
    """

    if eps is not None and sparse:
        raise ValueError("Cannot set 'eps' when sparse=True")

    classes, class_idx = np.unique(labels_true, return_inverse=True)
    clusters, cluster_idx = np.unique(labels_pred, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = sp.coo_matrix((np.ones(class_idx.shape[0]),
                                 (class_idx, cluster_idx)),
                                shape=(n_classes, n_clusters),
                                dtype=np.int)
    if sparse:
        contingency = contingency.tocsr()
        contingency.sum_duplicates()
    else:
        contingency = contingency.toarray()
        if eps is not None:
            # don't use += as contingency is integer
            contingency = contingency + eps
    return contingency


def getloopcount(savestr,null):

    # collect results from fold-wise run of GMM full covariance with LOOCV
    # JAN1_Svc_sallcluslabels_ctrl_fold1.pkl
    sets=['Tvc','Svc','TSvc','Tvc_tvc', 'Svc_svc','TSvc_tsvc', 'Tvct_s','Svcs_s','Tvct_Svcs_s','Tvct_tvc_s','Svcs_svc_s','Tvct_Svcs_tvc_svc_s']
    #sets=['Tc','Sc','TSc','Tc_tc','Sc_sc','TSc_tsc','Tct_s','Scs_s','Tct_Scs_s','Tct_tc_s','Scs_sc_s','Tct_Scs_tc_sc_s']
    
    import os
    import numpy as np
    ex=np.zeros(shape=[2,len(sets),16])
    for s in range(len(sets)):
        for fold in range(16):
            if null==0:
                if os.path.isfile(savestr  + str(sets[s]) + 'allcluslabels_fold' + str(fold) + '.pkl'):
                    ex[0,s,fold]=1
                if os.path.isfile(savestr  + str(sets[s]) + 'allcluslabels_ctrl_fold' + str(fold) + '.pkl'):
                    ex[1,s,fold]=1
            else:
                if os.path.isfile(savestr + str(sets[s]) + 'allcluslabels_null_fold' + str(fold) + '.pkl'):
                    ex[0,s,fold]=1
                if os.path.isfile(savestr  + str(sets[s]) + 'allcluslabels_null_ctrl_fold' + str(fold) + '.pkl'):
                    ex[1,s,fold]=1
    ex1=np.sum(ex,axis=2)
    print(ex1)
    return ex1
