import pickle
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

# outcome metrics I want:

# Level 1: LOOCV clustering
    # silhouette score
    # proportion of ambiguous clustering

# Level 2: aggregated LOOCV solutions in each subfold
    # silhouette score after cluster ensemble
    # silhouette score after new classification model
    # micro f1 for new classification model
    # macro f1 for new classification model
    # test set classification probability

# Level 3: aggregated subfold solutions in each mainfold
    # silhouette score
    # test set classification probability

# Level 4: commonality across mainfold solutions


# analysis run info:
#input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'
#modstr = '_mod_ctrl_'
input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/null/MDDnull/MDD__'
modstr = '_mod_null_'

sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s", "Tct_Scs_tc_sc_s"]
n_cv_folds = 4
n = 398
n_k = 8

do_level_1 = 0
do_level_2 = 1

pac_lvl1_done = 0

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

    silhouette_lvl1 = np.full([len(sets),n_cv_folds,n_cv_folds,n,n_k],np.nan)
    pac_lvl1 = np.full([len(sets),n_cv_folds,n_cv_folds,n_k],np.nan)
    for s in range(len(sets)):
        for mf in range(n_cv_folds):
            for sf in range(n_cv_folds):
                fold=(mf*n_cv_folds)+sf
                filestr=(input_filedir + sets[s] + modstr + str(fold))
                with open(filestr, "rb") as f:
                    mod = pickle.load(f)
                silhouette_lvl1[s,mf,sf,:,:]=mod.sil
                pac_lvl1[s, mf, sf, :] = mod.pac

    with open(input_filedir + modstr + 'sil_pac_lvl1.pkl', 'wb') as f:
        pickle.dump([silhouette_lvl1,pac_lvl1],f)



##################
#    LEVEL 2     #
##################

if do_level_2==1:

    silhouette1_lvl2 = np.full([len(sets),n_cv_folds,n_cv_folds, n_k],np.nan)
    silhouette2_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k], np.nan)
    microf1_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k,5], np.nan)
    macrof1_lvl2 = np.full([len(sets), n_cv_folds, n_cv_folds, n_k,5], np.nan)
    testproba_lvl2 = np.full([len(sets),n_cv_folds,n_cv_folds, n_k,5],np.nan)
    for s in range(len(sets)):
        for mf in range(n_cv_folds):
            for sf in range(n_cv_folds):
                fold=(mf*n_cv_folds)+sf
                filestr=(input_filedir + sets[s] + modstr + str(fold))
                with open(filestr, "rb") as f:
                    mod = pickle.load(f)
                silhouette1_lvl2[s,mf,sf,:]=mod.silhouette_cluster_ensembles

                microf1_lvl2[s, mf, sf, :, :] = mod.micro_f1
                macrof1_lvl2[s, mf, sf, :, :] = mod.macro_f1

                if hasattr(mod,'micro_f1'):
                    microf1_lvl2[s, mf, sf, :, :] = mod.micro_f1
                    macrof1_lvl2[s, mf, sf, :, :] = mod.macro_f1
                else:
                    print(s,mf,sf)

    fig = plt.figure(figsize=[20, 10])
    for s in range(len(sets)):
        plt.subplot(4, 3, s + 1);
        plt.title(sets[s])
        for mf in range(4):
            plt.plot([np.nanmean(silhouette1_lvl2[s, mf, :, k]) for k in range(n_k)])
    plt.show()

    fig = plt.figure(figsize=[20, 10])
    for s in range(len(sets)):
        plt.subplot(4, 3, s + 1);
        plt.title(sets[s])
        for mf in range(4):
            plt.plot([np.nanmean(microf1_lvl2[s, mf, :, k]) for k in range(n_k)])
    plt.show()

    fig = plt.figure(figsize=[20, 10])
    for s in range(len(sets)):
        plt.subplot(4, 3, s + 1);
        plt.title(sets[s])
        for mf in range(4):
            plt.plot([np.nanmean(macrof1_lvl2[s, mf, :, k]) for k in range(n_k)])
    plt.show()