import pickle
import numpy as np
import sys
from cv_clustering.agglom import doagglom_moremet
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')
from cv_clustering.mainfoldaggr import agglom, aggr4comp, agglom_best_k_per_sf, doagglomchks, agglomerrorwrap, \
    cross_sf_similarity_chk, samekagglom_error_mf, moreagglomloop,get_labels_rand
from sklearn.metrics import silhouette_samples, silhouette_score
from joblib import Parallel, delayed

#failed on /Volumes/ELEMENTS/clustering_pilot/clustering_output/IXI3_GMM/mod/IXI3_Sc_sc_mod_8

sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s", "Tct_Scs_tc_sc_s"]
setsize=np.array([82,82,150,84,84,154,82,82,150,84,84,154])

dattype=['MDD_GMM','MDD_GMM_null','MDD_spectral','IXI3_GMM','IXI_GMM_null','IXI_spectral','ALL_GMM']
modstr2use=['_mod_ctrl_','_mod_null_','_mod_','_mod_','_mod_null_','_mod_','_mod_']
pref=['normative_correction/FEB_','MDD__','MDD_spectral_','IXI3_','IXI2_','IXI2_spectral_', 'ALLALL3_']
N=[398,398,398,549,544,544,740]

ii=1
input_filedir = '/Volumes/ELEMENTS/clustering_pilot/clustering_output/' + dattype[ii] + '/mod/' + pref[ii]
modstr = modstr2use[ii]
n_cv_folds = 4
n = N[ii]
n_k = 8

# make sure par_sfnewclass.py is run first (using bb.py) so the hypergraph partitioning is done

do_level_1 = 0
do_level_2 = 0
do_level_3 = 0
do_level_35 = 0
do_level_30 = 0
do_level_labels=1

pac_lvl1_done = 0
reclass_lvl2_done = 0
subfold_similarity_done=0
mainfold_similarity_done=0
agglom_lvl3_done = 0
aggr00_lvl3_done = 0
agglom_by_k_done = 0
moveon=0
best_k_agglom_done=0

##################
#    LEVEL 1     #
##################

if do_level_1==1:
    print('doing level 1 for ' + (input_filedir))
    # calculate pac
    def calcpac(filestr):
        try:
            with open(filestr, "rb") as f:
                mod = pickle.load(f)
            if hasattr(mod,'pac'):
                pass
            else:
                mod.calc_consensus_matrix()
                mod.get_pac()
                with open(filestr, "wb") as f:
                    pickle.dump(mod,f)
        except:
            print('failed on ' + filestr)

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
        try:
            with open(filestr, "rb") as f:
                mod = pickle.load(f)
            if hasattr(mod,'testset_prob'):
                pass
            else:
                #mod.cluster_ensembles_new_classification()
                mod.sf_class_probas()
                with open(filestr, "wb") as f:
                    pickle.dump(mod,f)
        except:
            try:
                mod.cluster_ensembles()
                mod.cluster_ensembles_new_classification()
                mod.sf_class_probas()
                with open(filestr, "wb") as f:
                    pickle.dump(mod, f)
            except:
                print('failed on ' + filestr)


    if reclass_lvl2_done==0:
        for s in range(len(sets)):
            print(sets[s])
            for fold in range(16):
                calcreclass((input_filedir + sets[s] + modstr + str(fold)))


    clussizeedges = np.arange(0, n, 10)
    silhouette1_lvl2 = np.full([len(sets),n_cv_folds,n_cv_folds, n_k],np.nan) # for whole training set after cluster ensembles aggregation of loocv
    silhouette2_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan) # for test set after they're assigned using the new classifier
    microf1_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan)
    macrof1_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan)
    testproba_lvl2 = np.full([len(sets),n_cv_folds,n_cv_folds, n_k],np.nan)
    certedges = np.arange(0,1.1,.1)
    testprobadist_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k, 10], np.nan)

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
                    testprobadist_lvl2[s, mf, sf, k,:],b=np.histogram(mod.testset_prob[:,k], certedges)


    with open(input_filedir + modstr + 'sil_f1_prob_lvl2.pkl', 'wb') as f:
        pickle.dump([silhouette1_lvl2,silhouette2_lvl2,microf1_lvl2,macrof1_lvl2,testproba_lvl2,testprobadist_lvl2,clussize_CE_lvl2,clussize_test_lvl2],f)



##################
#    LEVEL 3     #
##################

if do_level_3 == 1:
    if agglom_lvl3_done == 0:
        #agglom(input_filedir, modstr, sets, n_k, n_cv_folds)
        allseterrors=Parallel(n_jobs=12)(delayed(moreagglomloop)([s, input_filedir, modstr]) for s in range(12))
        with open(input_filedir + modstr + 'allseterror_mf.pkl', 'wb') as f:
            pickle.dump([allseterrors], f)

    if aggr00_lvl3_done == 0:
        aggr4comp(input_filedir, modstr, sets, n_k, n_cv_folds)

    if agglom_by_k_done==0:
        agglomerrorwrap(input_filedir, modstr, n_cv_folds, sets, setsize, n_k)

    if subfold_similarity_done==0:
        cross_sf_similarity_chk(input_filedir, modstr, sets, setsize, n_k=8, n_cv_folds=4)

    if mainfold_similarity_done==0:
        samekagglom_error_mf(input_filedir, sets, setsize, n_k, n_cv_folds)

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
                        [X,clustering,assig,allbetas, tmp_testproba, argmaxYdot, argmaxYprob] = pickle.load(f)
                    silhouette_dot_lvl3[s,mf,k] = silhouette_score(mod.data[maintest], argmaxYdot)
                    silhouette_prob_lvl3[s, mf, k] = silhouette_score(mod.data[maintest], argmaxYprob)
                    topprob = np.array([np.max(tmp_testproba[i,:]) for i in range(tmp_testproba.shape[0])])
                    testproba_best_lvl3[s,mf,k] = np.nanmean(topprob)
                    for c in range(k+2):
                        testproba_best_perclus_lvl3[s,mf,k,c]=np.nanmean(topprob[np.where(argmaxYprob==c)[0]])
                        clussize_dot_lvl3[s,mf,k,c]=len(np.where(argmaxYdot==c)[0])
                        clussize_prob_lvl3[s,mf,k,c]=len(np.where(argmaxYprob==c)[0])

        with open((input_filedir + modstr + 'sil_prob_lvl3.pkl'), 'wb') as f:
            pickle.dump( [silhouette_dot_lvl3,silhouette_prob_lvl3,testproba_best_lvl3,testproba_best_perclus_lvl3,clussize_dot_lvl3,clussize_prob_lvl3], f)


####################
#    LEVEL 3.5     #
####################
if do_level_35==1:
    if best_k_agglom_done==0:
        agglom_best_k_per_sf(input_filedir, modstr, input_filedir_null, modstr_null, sets, setsize, n_cv_folds, n)
        with open((input_filedir + 'aggr_betas_best_sf_k.pkl'), 'rb') as f:
            [bestk_mdd,final_k,allbetas_sfk,silmaintrain,silmaintest,train_clus_assig,test_clus_assig,train_clus_prob,test_clus_prob,train_clus_assig25,test_clus_assig25] = pickle.load(f)


################################################
#    LEVEL 3.0 -- agglom with more metrics     #
################################################
if do_level_30==1:
    averageerror=np.full([12,4,n_k,n_k],np.nan)
    n_vecs = np.full([12,4,n_k,n_k, n_k+2],np.nan)
    source_vecs = np.full([12,4,n_k, n_k, n_k + 2], np.nan)
    ed_btw_vecs = np.full([12,4,n_k, n_k, n_k + 2], np.nan)
    avgclus=[None]*len(sets)
    for s in range(len(sets)):
        filepath2use = (input_filedir + sets[s] + modstr)
        avgclus[s]=[None]*n_cv_folds
        for mf in range(n_cv_folds):
            averageerror[s,mf,:,:], n_vecs[s,mf,:,:,:], source_vecs[s,mf,:,:,:], ed_btw_vecs[s,mf,:,:,:], avgclus[s][mf] = doagglom_moremet(0, mf ,n_cv_folds, setsize[s], n_k, filepath2use)

    with open((input_filedir + 'agglom_moremet.pkl'), 'wb') as f:
        pickle.dump([averageerror,n_vecs,source_vecs,ed_btw_vecs,avgclus], f)


if do_level_labels==1:
    print(dattype[ii])
    print(input_filedir)
    allsol = np.full([8, n, 12], np.nan)
    allrand = np.full([12, 12, 8], np.nan)
    for k in range(8):
        print(k)
        allsol[k, :, :], allrand[:, :, k] = get_labels_rand(ii, k + 2, n)
    with open((input_filedir + 'allsol_labels.pkl'), 'wb') as f:
        pickle.dump([allsol,allrand], f)