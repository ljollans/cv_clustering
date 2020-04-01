# load LOOCV cluster assignments and get a consensus cluster naming
import sys
import os
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy import sparse as sp

from utils import select_trainset, percent_overlap_vectors, get_pac


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


def get_clusassignments_from_LOOCV(set,mainfold,subfold):

    fold_number=(4*mainfold)+subfold

    sets = [ 'Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct_s', 'Svcs_s', 'Tvct_Svcs_s', 'Tvct_tvc_s',
             'Svcs_svc_s', 'Tvct_Svcs_tvc_svc_s' ]
    savedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'

    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open ( (cv_assignment_dir + "CVassig398.csv"), "r" ) as f:
        reader = csv.reader ( f, delimiter="," )
        cv_assignment = np.array ( list ( reader ) ).astype ( float )

    pkl_filename = (savedir + sets[ set ] + 'allcluslabels_fold' + str (fold_number) + '.pkl')
    with open ( pkl_filename, "rb" ) as file:
        Aorig = pickle.load ( file )
    trainset = select_trainset ( cv_assignment, mainfold, subfold )
    A1 = Aorig[ trainset, :, : ]
    A2 = A1[ :, :, trainset ]
    return A2


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


def n_clus_retrieval_chk(cluster_assignments):
    n_maxcluster = np.zeros(shape=[cluster_assignments.shape[1]])
    pacs = np.zeros(shape=[cluster_assignments.shape[1]])
    for ngroups in range(cluster_assignments.shape[1]):
        cluster_assignments_k = cluster_assignments[:, ngroups, :]
        co_cluster_count = get_co_cluster_count(cluster_assignments_k)
        try:
            pacs[ngroups] = get_pac(co_cluster_count)
        except:
            pacs[ngroups] = 0.0

        maxmatches, findmax = get_maxmatches(cluster_assignments_k, co_cluster_count, 1 - 0.2)

        final_assignment = get_final_assignment(maxmatches, 25)

        n_maxcluster[ngroups] = (len(final_assignment))
    u_cr = np.unique(n_maxcluster)
    cr_count = [len(np.where(n_maxcluster == i)[0]) for i in u_cr]
    max_cr = np.max(cr_count)
    find_max_cr = np.where(cr_count == max_cr)

    return n_maxcluster, pacs, u_cr[find_max_cr[0][0]]


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
    fig = plt.figure (); plt.rc('font', size=8)
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
        plt.title ( 'number of unique clusters across LOOCV' )
        plt.plot ( cr ); plt.xlabel('k');
        plt.xticks(np.arange(len(cr)), np.arange(len(cr))+2)
        ctr += 1;
        plt.subplot ( 4, 3, ctr );
        plt.title ( 'proportion of ambiguous clustering' )
        plt.plot ( pac );plt.xlabel('k');
        plt.xticks(np.arange(len(cr)), np.arange(len(cr))+2)
    plt.subplots_adjust(0.125,0.1,0.9,0.9,0.1,0.45)
    plt.show ()



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


def get_co_cluster_count(cluster_assignments):
    # returns ratio of co_clustering for every pair of observations
    # input: cluster_assignment (i x N matrix) for i iterations and N observations
    # output: co_cluster_count (NxN matrix) with values between
    # 0 (never cluster together) and 1 (always cluster together)

    # e.g. co_cluster_count[n1,n2] is the ratio of times cluster_assignments[i,n1]==cluster_assignment[i,n2]) within
    # every instance of n1 and n2 being assigned a group in the same i


    co_cluster_count = np.full([cluster_assignments.shape[1], cluster_assignments.shape[1]], 0.00)
    for n1 in range(cluster_assignments.shape[1]):
        for n2 in range(cluster_assignments.shape[1]):
            if n1 > n2:
                co_cluster_sum = len(np.where(cluster_assignments[:, n1] == cluster_assignments[:, n2])[0])
                co_assignment_instances = cluster_assignments.shape[1] - len(
                    np.unique(
                        np.append(
                            np.where(np.isnan(cluster_assignments[:, n1]))[0],
                            np.where(np.isnan(cluster_assignments[:, n2]))[0],
                        )
                    )
                )
                co_cluster_count[n1, n2] = co_cluster_sum / co_assignment_instances

    return co_cluster_count


def get_maxmatches(cluster_assignments, co_cluster_count, ratio):
    # returns assignments for pairs of observations in cluster_assignments where ratio of
    # co_clustering is >= ratio
    # input: cluster_assignment (i x N matrix) for i iterations and N observations,
    #        co_cluster_count (NxN matrix) with values between 0 and 1 (always cluster together)
    #        ratio: value between 0 and 1 for the minimum ratio of co-clustering for assignments to be returned
    # output: findmax: indices of f observation pairs that co-cluster >= ratio
    #         maxmatches (i x f) matrix containing the overlapping assignments for the pairs in findmax

    max_match = ratio  # np.max(co_cluster_count) * ratio_of_max_as_lower
    findmax = np.where(co_cluster_count >= max_match)
    print(
        "highest match frequency=",
        str(max_match),
        "occurring for",
        str(len(findmax[0])),
        "pairs.",
    )
    maxmatches = np.full([cluster_assignments.shape[0], len(findmax[0]), ], np.nan)
    for n in range(len(findmax[0])):
        a1 = findmax[0][n]
        a2 = findmax[1][n]
        matches = np.where(cluster_assignments[:, a1] == cluster_assignments[:, a2])[0]
        maxmatches[matches, n] = cluster_assignments[matches, a1]
    return maxmatches, findmax


def get_final_assignment(maxmatches, max_pct):
    # for the assignments in maxmatches identifies the number of unique assignment patterns
    # if the overlap of an assignment pattern with any already identified pattern is less than ratio
    # then the pattern is treated as a new unique assignment pattern
    # input: maxmatches:  (i x f) matrix containing the f assignments from i iterations to be evaluated
    #        max_pct: the maximum perecentage overlap an assignment pattern can have with another to still be
    #                 considered unique
    # output: final_assignment: (i x u) matrix with the u unique assignment patterns

    final_assignment = []
    for n in range(maxmatches.shape[1]):
        if len(final_assignment) > 0:
            tmp_match_pct = np.zeros(len(final_assignment))
            for clus in range(len(final_assignment)):
                tmp_match_pct[clus] = percent_overlap_vectors(
                    maxmatches[:, n], final_assignment[clus]
                )
            if np.max(tmp_match_pct) < max_pct:
                final_assignment.append(maxmatches[:, n])
        else:
            final_assignment.append(maxmatches[:, n])
    return final_assignment


def match_assignments_to_final_assignments(cluster_assignments, final_assignment):
    match_pct = np.zeros(shape=[cluster_assignments.shape[0],len(final_assignment)])
    for n in range(cluster_assignments.shape[0]):
        for clus in range(len(final_assignment)):
            match_pct[n,clus] = percent_overlap_vectors(
                cluster_assignments[:,n], final_assignment[clus]
            )
    return match_pct