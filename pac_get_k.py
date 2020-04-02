from loocv_assigmatcher_nov import getloopcount, get_k_from_bic, get_clusassignments_from_LOOCV, n_clus_retrieval_chk
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs1.pkl'
    with open(pkl_filename, "wb") as file:
        pickle.dump(all_pac_diffs1, file)

    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs2.pkl'
    with open(pkl_filename, "wb") as file:
        pickle.dump(all_pac_diffs2, file)

if make_pandas_df==1:
    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs1.pkl'
    with open(pkl_filename, "rb") as file:
        all_pac_diffs1= pickle.load(file)

    pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs2.pkl'
    with open(pkl_filename, "rb") as file:
        all_pac_diffs2= pickle.load(file)
        
    df=pd.DataFrame({}, columns=['pacdiff', 'set', 'mainfold','subfold', 'k', 'ctr','ventricles'])     
    for set in range(len(sets_null)):
        for mainfold in range(4):
            for subfold in range(4):
                for ctr in range(2):
                    for k in range(8):
                        tmp_df = pd.DataFrame(
                            {'pacdiff': all_pac_diffs1[set,mainfold,subfold,k,ctr],
                             'set':[set],
                             'mainfold':[mainfold],
                             'subfold':[subfold],
                             'k':[k],
                             'ctr':[ctr],
                             'ventricles':[1],
                             },
                            columns=['pacdiff', 'set', 'mainfold','subfold', 'k', 'ctr','ventricles'])
                        df = df.append(tmp_df)
                        tmp_df = pd.DataFrame(
                            {'pacdiff': all_pac_diffs2[set,mainfold,subfold,k,ctr],
                             'set':[set],
                             'mainfold':[mainfold],
                             'subfold':[subfold],
                             'k':[k],
                             'ctr':[ctr],
                             'ventricles':[0],
                             },
                            columns=['pacdiff', 'set', 'mainfold','subfold', 'k', 'ctr','ventricles'])
                        df = df.append(tmp_df)
    df.to_csv(r'/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs.csv', index = False)     

df = pd.read_csv("/Users/lee_jollans/Projects/clustering_pilot/all_pac_diffs.csv") 
fig=plt.figure()    
for set in range(len(sets_null)):
    plt.subplot(4,3,set+1)
    ax = sns.lineplot(x="k",y="pacdiff",hue="ventricles",data=df[df["set"]==set])
plt.show()
        