from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.stats import ttest_ind
from cv_clustering.loocv_assigmatcher_nov import getloopcount, get_k_from_bic, get_clusassignments_from_LOOCV, n_clus_retrieval_chk, \
    get_consensus_labels, return_train_data
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv

from cv_clustering.multi_logr_bag import multi_logr_bagr
from cv_clustering.utils import select_trainset, rand_score_withnans, get_gradient_change

savedir_null = '/Users/lee_jollans/Projects/clustering_pilot/null/JAN1_'
savedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'

sets_null = ['Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct', 'Svcs', 'Tvct_Svcs', 'Tvct_tvc',
             'Svcs_svc', 'Tvct_Svcs_tvc_svc']
sets1 = ['Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct_s', 'Svcs_s', 'Tvct_Svcs_s', 'Tvct_tvc_s',
         'Svcs_svc_s', 'Tvct_Svcs_tvc_svc_s']
sets2 = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
         'Scs_sc_s', 'Tct_Scs_tc_sc_s']


calculate_pac_diffs=0
make_pandas_df=0
plot_pacdiffs=1
calculate_consensus_labels_set2=0
calculate_consensus_labels_set1=0
calculate_consensus_labels_set_null=0
calculate_rand_across_folds=0
calculate_consensus_silhouette_set2=0
calculate_consensus_davies_bouldin_set2=0
plot_all_F1s=0
build_new_regression_consensus_labels=0
transform_pac_gradient_change=0

def do_plots_scores_by_sets(score,title):
    fig=plt.figure()
    plt.imshow(score); plt.colorbar()
    plt.xticks(np.arange(8),np.arange(8)+2)
    plt.yticks(np.arange(len(sets2)),sets2)
    plt.title(title)
    plt.show()
    
    fig=plt.figure(figsize=[10,10])
    for s in range(3):
        plt.subplot(3,1,s+1)
        a=np.arange(0,11,3)+s
        plt.plot(score[a.astype(int)].T)
        ll=[sets2[a[i].astype(int)] for i in range(4)]
        plt.legend(ll)
        plt.xticks(np.arange(8),np.arange(8)+2)
        plt.xlabel('k')
        plt.ylabel(title)
    plt.show()

if calculate_pac_diffs==1:
    ex_null = getloopcount(savedir_null, 1, sets_null)
    ex_1 = getloopcount(savedir, 0, sets1)
    ex_2 = getloopcount(savedir, 0, sets2)

    [best_sil_null, best_cal_null, best_bic_null, best_k_mf_null, best_k_sf_null,
     best_k_loocv_null, sil_null, cal_null, bic_null, pct_agreement_k_null] \
        = get_k_from_bic(savedir_null, ex_null, 1, sets_null)

    [best_sil_1, best_cal_1, best_bic_1, best_k_mf_1, best_k_sf_1,
     best_k_loocv_1, sil_1, cal_1, bic_1, pct_agreement_k_1] \
        = get_k_from_bic(savedir, ex_1, 0, sets1)

    [best_sil_2, best_cal_2, best_bic_2, best_k_mf_2, best_k_sf_2,
     best_k_loocv_2, sil_2, cal_2, bic_2, pct_agreement_k_2] \
        = get_k_from_bic(savedir, ex_2, 0, sets2)

    all_pacs1 = np.full([len(sets_null), 4, 4, 8, 2], np.nan)
    all_pacs2 = np.full([len(sets_null), 4, 4, 8, 2], np.nan)
    all_pacs_null = np.full([len(sets_null), 4, 4, 8], np.nan)

    all_ucn1 = np.full([len(sets_null), 4, 4, 8, 2], np.nan)
    all_ucn2 = np.full([len(sets_null), 4, 4, 8, 2], np.nan)
    all_ucn_null = np.full([len(sets_null), 4, 4, 8], np.nan)

    all_pac_diffs1 = np.full([len(sets_null), 4, 4, 8, 2], np.nan)
    all_pac_diffs2 = np.full([len(sets_null), 4, 4, 8, 2], np.nan)
    for set in range(len(sets_null)):
        for mainfold in range(4):
            for subfold in range(4):
                cluster_assignments10 = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets1, savedir, 0, 0)
                cluster_assignments11 = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets1, savedir, 0, 1)
                cluster_assignments20 = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets2, savedir, 0, 0)
                cluster_assignments21 = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets2, savedir, 0, 1)
                cluster_assignments_null = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets_null, savedir_null,
                                                                          1, 0)

                cr10, pac10, plateau_cr10 = n_clus_retrieval_chk(cluster_assignments10, .7, 25)
                cr11, pac11, plateau_cr11 = n_clus_retrieval_chk(cluster_assignments11, .7, 25)
                cr20, pac20, plateau_cr20 = n_clus_retrieval_chk(cluster_assignments20, .7, 25)
                cr21, pac21, plateau_cr21 = n_clus_retrieval_chk(cluster_assignments21, .7, 25)

                cr_null, pac_null, plateau_cr_null = n_clus_retrieval_chk(cluster_assignments_null, .7, 25)

                all_pac_diffs1[set, mainfold, subfold, :, 0] = pac_null - pac10
                all_pac_diffs1[set, mainfold, subfold, :, 1] = pac_null - pac11
                all_pac_diffs2[set, mainfold, subfold, :, 0] = pac_null - pac20
                all_pac_diffs2[set, mainfold, subfold, :, 1] = pac_null - pac21

                all_pacs1[set, mainfold, subfold, :, 0]=pac10
                all_pacs1[set, mainfold, subfold, :, 1] = pac11
                all_pacs2[set, mainfold, subfold, :, 0] = pac20
                all_pacs2[set, mainfold, subfold, :, 1] = pac21
                all_pacs_null[set, mainfold, subfold, :] = pac_null

                all_ucn1[set, mainfold, subfold, :, 0] = cr10
                all_ucn1[set, mainfold, subfold, :, 1] = cr11
                all_ucn2[set, mainfold, subfold, :, 0] = cr20
                all_ucn2[set, mainfold, subfold, :, 1] = cr21
                all_ucn_null[set, mainfold, subfold, :] = cr_null

    dumpthese = ["all_pac_diffs1","all_pac_diffs2","all_pacs1","all_pacs2","all_pacs_null","all_ucn1","all_ucn2","all_ucn_null"]
    for d in dumpthese:
        pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/' + d + '.pkl'
        with open(pkl_filename, "wb") as file:
            eval("pickle.dump(" + d + ", file)")

if transform_pac_gradient_change==1:
    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs2.pkl'
    with open(pkl_filename, "rb") as file:
        all_pacs2 = pickle.load(file)
    print(all_pacs2.shape)
    pac_gradient=np.full(all_pacs2.shape, np.nan)
    for set in range(len(sets_null)):
        for mainfold in range(4):
            for subfold in range(4):
                for ctr in range(2):
                    pac_gradient[set,mainfold,subfold,:,ctr] = get_gradient_change(all_pacs2[set,mainfold,subfold,:,ctr])

if make_pandas_df==1:
    dumpthese = ["all_pac_diffs1", "all_pac_diffs2", "all_pacs1", "all_pacs2", "all_pacs_null", "all_ucn1", "all_ucn2",
                 "all_ucn_null"]
    for d in dumpthese:
        pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/' + d + '.pkl'
        with open(pkl_filename, "rb") as file:
            exec(d +  " = pickle.load(file)")
        
    df=pd.DataFrame({}, columns=['pac', 'pacdiff', 'set', 'mainfold','subfold', 'k', 'ctr','ventricles'])
    for set in range(len(sets_null)):
        for mainfold in range(4):
            for subfold in range(4):
                for ctr in range(2):
                    for k in range(8):
                        tmp_df = pd.DataFrame(
                            {'pac': all_pacs1[set,mainfold,subfold,k,ctr],
                             'pacdiff': all_pac_diffs1[set,mainfold,subfold,k,ctr],
                             'set':[set],
                             'mainfold':[mainfold],
                             'subfold':[subfold],
                             'k':[k+2],
                             'ctr':[ctr],
                             'ventricles':[1],
                             },
                            columns=['pac','pacdiff', 'set', 'mainfold','subfold', 'k', 'ctr','ventricles'])
                        df = df.append(tmp_df)
                        tmp_df = pd.DataFrame(
                            {'pac': all_pacs2[set,mainfold,subfold,k,ctr],
                             'pacdiff': all_pac_diffs2[set,mainfold,subfold,k,ctr],
                             'set':[set],
                             'mainfold':[mainfold],
                             'subfold':[subfold],
                             'k':[k+2],
                             'ctr':[ctr],
                             'ventricles':[0],
                             },
                            columns=['pac','pacdiff', 'set', 'mainfold','subfold', 'k', 'ctr','ventricles'])
                        df = df.append(tmp_df)
    df.to_csv(r'/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs.csv', index = False)

if plot_pacdiffs==1:
    df = pd.read_csv("/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs.csv")
    fig=plt.figure(figsize=[15,10])
    for set in range(len(sets_null)):
        plt.subplot(4,3,set+1)
        plt.title(sets2[set])
        ax = sns.lineplot(x="k",y="pac",hue="ctr",data=df[(df["set"]==set) & (df["ventricles"]==1)])
    plt.show()

if pac_ttests==1:
    sets2 = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
             'Scs_sc_s', 'Tct_Scs_tc_sc_s']
    incl_cov=[0,0,0,1,1,1,0,0,0,1,1,1]
    resid_globals=[0,0,0,0,0,0,1,1,1,1,1,1]
    modality = [0, 1,2,0,1,2,0,1,2,0,1,2]

    df['incl_cov'] = df['set']
    df['resid_globals'] = df['set']
    df['modality'] = df['set']
    for set in range(len(sets2)):
        df['incl_cov'].replace(set,incl_cov[set], inplace=True)
        df['resid_globals'].replace(set, resid_globals[set], inplace=True)
        df['modality'].replace(set, modality[set], inplace=True)

    ttest_ind(df['pac'][df['incl_cov']==0],df['pac'][df['incl_cov']==1])

    #ttest_ind(df['pac'][df['incl_cov']==0],df['pac'][df['incl_cov']==1])
    #Ttest_indResult(statistic=1.9351553611611025, pvalue=0.05301708489456915)
    #ttest_ind(df['pac'][df['resid_globals']==0],df['pac'][df['resid_globals']==1])
    #Ttest_indResult(statistic=-52.62744503284114, pvalue=0.0)

if calculate_consensus_labels_set2==1:
    savedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)

    consensus_labels = np.full([len(sets2),2,4,4,8,398],np.nan)
    for mainfold in range(4):
        for subfold in range(4):
            trainset=select_trainset(cv_assignment, mainfold, subfold)
            for set in range(len(sets_null)):
                for ctr in range(2):
                    A = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets2, savedir, 0, ctr)
                    for k in range(8):
                        A2=A[:,k,:]
                        try:
                            consensus_labels[set,ctr,mainfold,subfold,k,trainset] = get_consensus_labels(A2, k+2, 0)
                        except:
                            print(mainfold,subfold,set,ctr,k)


    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels.pkl'
    with open(pkl_filename, "wb") as file:
        pickle.dump(consensus_labels, file)
        
if calculate_consensus_labels_set1==1:
    savedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_'
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)

    consensus_labels = np.full([len(sets2),2,4,4,8,398],np.nan)
    for mainfold in range(4):
        for subfold in range(4):
            trainset=select_trainset(cv_assignment, mainfold, subfold)
            for set in range(len(sets_null)):
                for ctr in range(2):
                    A = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets1, savedir, 0, ctr)
                    for k in range(8):
                        A2=A[:,k,:]
                        try:
                            consensus_labels[set,ctr,mainfold,subfold,k,trainset] = get_consensus_labels(A2, k+2, 0)
                        except:
                            print(mainfold,subfold,set,ctr,k)


    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels_with_ventricles.pkl'
    with open(pkl_filename, "wb") as file:
        pickle.dump(consensus_labels, file)

if calculate_consensus_labels_set_null==1:

    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)

    consensus_labels = np.full([len(sets2),2,4,4,8,398],np.nan)
    for mainfold in range(4):
        for subfold in range(4):
            trainset=select_trainset(cv_assignment, mainfold, subfold)
            for set in range(len(sets_null)):
                for ctr in range(2):
                    A = get_clusassignments_from_LOOCV(set, mainfold, subfold, sets_null, savedir_null, 1, ctr)
                    for k in range(8):
                        A2=A[:,k,:]
                        try:
                            consensus_labels[set,ctr,mainfold,subfold,k,trainset] = get_consensus_labels(A2, k+2, 0)
                        except:
                            print(mainfold,subfold,set,ctr,k)


    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels_null.pkl'
    with open(pkl_filename, "wb") as file:
        pickle.dump(consensus_labels, file)
        
if calculate_rand_across_folds==1:
    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels.pkl'
    with open(pkl_filename, "rb") as file:
        consensus_labels= pickle.load(file)
    rr=np.full([len(sets_null),2,8,4],np.nan)
    for set in range(len(sets_null)):
        for ctr in range(2):
            for k in range(8):
                for mainfold in range(4):
                    tmp_rand = np.full([4,4],np.nan)
                    for sf1 in range(4):
                        for sf2 in range(4):
                            if sf2<sf1:
                                tmp_rand[sf1,sf2]=rand_score_withnans(consensus_labels[set,ctr,mainfold,sf1,k,:],consensus_labels[set,ctr,mainfold,sf2,k,:])
                    rr[set,ctr,k,mainfold]=np.nanmean(tmp_rand)
        
    do_plots_scores_by_sets(np.nanmean(rr[:,0,:,:],axis=2),'rand score')

if calculate_consensus_silhouette_set2==1:
    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels.pkl'
    with open(pkl_filename, "rb") as file:
        consensus_labels= pickle.load(file)
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)
    sil = np.full([len(sets2),2,4,4,8],np.nan)
    for set in range(len(sets_null)):
        for ctr in range(2):
            if ctr==1:
                data_path = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/" + "MDD__" + sets2[set] + "_ctrl.csv"
            else:
                data_path = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/" + "MDD__" + sets2[set] + ".csv"
            with open(data_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                x = np.array(list(reader)).astype(float)
            for mainfold in range(4):
                for subfold in range(4):
                    trainset=select_trainset(cv_assignment, mainfold, subfold)
                    x2use=return_train_data(x, mainfold, subfold)
                    for k in range(8):
                        try:
                            sil[set,ctr,mainfold,subfold,k] = silhouette_score(x2use,consensus_labels[set,ctr,mainfold,subfold,k,trainset]+1)
                        except:
                            print('failed on k=',k,'for set/ctr',sets2[set],str,'mainfold/subfold',mainfold,subfold)
                            print(consensus_labels[set,ctr,mainfold,subfold,k,trainset])
    
    do_plots_scores_by_sets(np.nanmean(np.nanmean(sil[:,0,:,:,:],axis=2),axis=1),'silhouette score')
    
if calculate_consensus_davies_bouldin_set2==1:
    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels.pkl'
    with open(pkl_filename, "rb") as file:
        consensus_labels= pickle.load(file)
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)
    dbs = np.full([len(sets2),2,4,4,8],np.nan)
    for set in range(len(sets_null)):
        for ctr in range(2):
            if ctr==1:
                data_path = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/" + "MDD__" + sets2[set] + "_ctrl.csv"
            else:
                data_path = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/" + "MDD__" + sets2[set] + ".csv"
            with open(data_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                x = np.array(list(reader)).astype(float)
            for mainfold in range(4):
                for subfold in range(4):
                    trainset=select_trainset(cv_assignment, mainfold, subfold)
                    x2use=return_train_data(x, mainfold, subfold)
                    for k in range(8):
                        try:
                            dbs[set,ctr,mainfold,subfold,k] = davies_bouldin_score(x2use,consensus_labels[set,ctr,mainfold,subfold,k,trainset]+1)
                        except:
                            print('failed on k=',k,'for set/ctr',sets2[set],str,'mainfold/subfold',mainfold,subfold)
                            print(consensus_labels[set,ctr,mainfold,subfold,k,trainset])
    
    do_plots_scores_by_sets(np.nanmean(np.nanmean(dbs[:,0,:,:,:],axis=2),axis=1),'davies bouldin score')

if plot_all_F1s==1:
    all_f1s=np.full([len(sets2),2,4,4,8],np.nan)
    for set in range(len(sets_null)):
        for ctr in range(2):
            for mainfold in range(4):
                for subfold in range(4):
                    fold=(mainfold*4)+subfold
                    if ctr == 1:
                        pkl_filename = (savedir + sets2[set] + 'F1_ctrl_fold' + str(fold) + '.pkl')
                    else:
                        pkl_filename = (savedir + sets2[set] + 'F1_fold' + str(fold) + '.pkl')
                    with open(pkl_filename, "rb") as file:
                        F1 = pickle.load(file)
                    all_f1s[set,ctr,mainfold,subfold,:]=np.nanmean(F1,axis=0)
    do_plots_scores_by_sets(np.nanmean(np.nanmean(all_f1s[:,0,:,:,:],axis=2),axis=1),'F1 score')

if build_new_regression_consensus_labels==1:
    
    all_macro_f1=np.full([len(sets2),2,4,4,8],np.nan)
    all_micro_f1=np.full([len(sets2),2,4,4,8],np.nan)
    
    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_consensuslabels.pkl'
    with open(pkl_filename, "rb") as file:
        consensus_labels = pickle.load(file)
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)

    for set in [0]:#range(len(sets_null)):
        for ctr in range(2):
            if ctr == 1:
                data_path = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/" + "MDD__" + sets2[
                    set] + "_ctrl.csv"
            else:
                data_path = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/" + "MDD__" + sets2[
                    set] + ".csv"
            with open(data_path, "r") as f:
                reader = csv.reader(f, delimiter=",")
                x = np.array(list(reader)).astype(float)
            for mainfold in range(4):
                for subfold in range(4):
                    trainset = select_trainset(cv_assignment, mainfold, subfold)
                    x2use = return_train_data(x, mainfold, subfold)
                    for k in range(8):
                        truth=consensus_labels[set,ctr,mainfold,subfold,k,trainset]
                        try:
                            print(np.max(truth)+1)
                            (
                                auc_across_cv_folds,
                                f1_across_cv_folds,
                                beta_avg_across_folds,
                                overall_prediction_continuous,
                                overall_prediction_discrete,
                                auc_per_cv_fold,
                                f1_per_cv_fold,
                                betas_per_fold,
                                n_across_cv_folds
                            ) = multi_logr_bagr(25, np.append(np.expand_dims(truth,axis=1),x2use,axis=1), (np.max(truth)+1.00).astype(int), 10, 0)
                            macroF1=np.nanmean(f1_across_cv_folds)
                            microF1=0
                            for g in range(len(np.unique(truth))):
                                microF1+=f1_across_cv_folds[g]*len(np.where(truth==g)[0])
                            microF1=microF1/len((truth))
                            all_macro_f1[set,ctr,mainfold,subfold,k]=macroF1
                            all_micro_f1[set,ctr,mainfold,subfold,k]=microF1
                        except:
                            print(truth)
                            raise Exception()
