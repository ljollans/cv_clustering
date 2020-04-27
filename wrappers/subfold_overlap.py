import numpy as np
import pickle
import csv
from cv_clustering.loocv_assigmatcher_nov import get_co_cluster_count
from cv_clustering.utils import get_pac, coph_cor
from sklearn.metrics import silhouette_score

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/ALL/wspecsamp_"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/ALL/wspecsamp_')

# presets
sets = [
    "Tc",
    "Sc",
    "TSc",
    "Tc_tc",
    "Sc_sc",
    "TSc_tsc",
    "Tct_s",
    "Scs_s",
    "Tct_Scs_s",
    "Tct_tc_s",
    "Scs_sc_s",
    "Tct_Scs_tc_sc_s",
]

with open((cv_assignment_dir + "CVassig740.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

microF1 = np.full([len(sets),8,4,4],np.nan)
macroF1 = np.full([len(sets),8,4,4],np.nan)
loocvpac = np.full([len(sets),8,4,4],np.nan)
all_labels = np.full([740,8,len(sets),4,4],np.nan)
pac = np.full([len(sets),8,4],np.nan)
coph = np.full([len(sets),8,4],np.nan)

for current_set in range(len(sets)):
    for mainfold in range(4):
        for subfold in range(4):
            fold = (mainfold*4)+subfold

            pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
            print(pkl_filename)
            with open(pkl_filename, "rb") as file:
                mod = pickle.load(file)
            mod.calc_consensus_matrix()
            mod.get_pac()
            with open(pkl_filename, "wb") as file:
                pickle.dump(mod,file)

            microF1[current_set, :, mainfold, subfold] = np.nanmean(mod.micro_f1, axis=1)
            macroF1[current_set, :, mainfold, subfold] = np.nanmean(mod.macro_f1, axis=1)
            loocvpac[current_set, :, mainfold, subfold] = mod.pac

for current_set in range(len(sets)):
    for k in range(8):
        for mainfold in range(4):
            for subfold in range(4):
                fold = (mainfold*4)+subfold

                pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
                print(pkl_filename)
                with open(pkl_filename, "rb") as file:
                    mod = pickle.load(file)

                i=np.nanmean(mod.allitcpt[k], axis=1)
                if k>0:
                    b=np.nanmean(mod.allbetas[k],axis=2)
                    bi = np.append(np.expand_dims(i, axis=1).T, b, axis=0)
                else:
                    b = np.nanmean(mod.allbetas[k], axis=1)
                    bi = np.append(np.expand_dims(i[0],axis=1).T,b,axis=0)

                tmpX = np.append(np.ones(shape=[mod.data.shape[0], 1]), mod.data, axis=1)
                newY = tmpX.dot(bi)
                if k==0:
                    newY = np.expand_dims(newY, axis=1)
                    newY = np.append(newY,1-newY, axis=1)
                argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
                all_labels[:,k,current_set,mainfold,subfold] = argmaxY

            maintrain = np.where(np.isfinite(cv_assignment[:, 0]))[0]

            crit = all_labels[:, k, current_set, mainfold, :].T
            A = crit[:, maintrain]

            consensus_matrix = get_co_cluster_count(A)
            pac[current_set, k, mainfold] = get_pac(consensus_matrix)
            coph[current_set, k, mainfold] = coph_cor(consensus_matrix)

pkl_filename = (savedir + 'all_labels.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(all_labels, file)
pkl_filename = (savedir + 'all_pac.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(pac, file)
pkl_filename = (savedir + 'all_coph.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(coph, file)
pkl_filename = (savedir + 'all_microf1.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(microF1, file)
pkl_filename = (savedir + 'all_macrof1.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(macroF1, file)
pkl_filename = (savedir + 'all_loocvpac.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(loocvpac, file)


sil = np.full([len(sets),8,4,3],np.nan)
allargmax = np.full([740,12,8,4],np.nan)
allproba = np.full([740,12,8,4],np.nan)
for current_set in range(len(sets)):
    print(sets[current_set])
    data_path = input_files_dir + sets[current_set] + "_ctrl.csv"
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)
    for k in range(8):
        for mainfold in range(4):

            ss = (savedir + sets[current_set] + '_aggregated_betas_k' + str(k) + '_' + str(mainfold) + '.csv')
            with open(ss, "r") as f:
                reader = csv.reader(f, delimiter=",")
                aggregated_betas = np.array(list(reader)).astype(float)

            tmpX = np.append(np.ones(shape=[data.shape[0], 1]), data, axis=1)
            newY = tmpX.dot(aggregated_betas)
            argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
            valmaxY = np.array([np.max(newY[i, :]) for i in range(newY.shape[0])])

            allargmax[:,current_set,k,mainfold]=argmaxY
            allproba[:, current_set, k, mainfold] = valmaxY

            # silhouette score for all
            sil[current_set,k,mainfold,0]=silhouette_score(data,argmaxY)
            # silhouette score for train
            maintrain = np.where(np.isfinite(cv_assignment[:, 0]))[0]
            sil[current_set, k, mainfold, 1] = silhouette_score(data[maintrain,:], argmaxY[maintrain])
            # silhouette score for test
            maintest = np.where(np.isnan(cv_assignment[:, 0]))[0]
            sil[current_set, k, mainfold, 2] = silhouette_score(data[maintest,:], argmaxY[maintest])

pkl_filename = (savedir + 'all_sil.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(sil, file)
pkl_filename = (savedir + 'all_labelsmain.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(allargmax, file)
pkl_filename = (savedir + 'all_probamain.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(allproba, file)



