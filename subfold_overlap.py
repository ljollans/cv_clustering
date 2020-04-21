import numpy as np
import pickle

from loocv_assigmatcher_nov import get_co_cluster_count
from utils import contingency_matrix, get_pac, coph_cor

from beta_aggregate import aggregate

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_')

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


all_labels = np.full([398,8,len(sets),2,4,4],np.nan)

for current_set in range(len(sets)):
    for ctr in range(2):
        for k in range(8):
            for mainfold in range(4):
                for subfold in range(4):
                    fold = (mainfold*4)+subfold

                    if ctr==1:
                        pkl_filename = (savedir + sets[current_set] + '_mod_ctrl_' + str(fold))
                    else:
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
                    all_labels[:,k,current_set,ctr,mainfold,subfold] = argmaxY

pkl_filename = (savedir + 'all_labels.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(all_labels, file)

import csv
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

pac = np.full([len(sets),2,8,4],np.nan)
coph = np.full([len(sets),2,8,4],np.nan)

for current_set in range(len(sets)):
    for ctr in range(2):
        for k in range(8):
            for mainfold in range(4):

                maintrain = np.where(np.isfinite(cv_assignment[:, 0]))[0]

                crit = all_labels[:,k,current_set,ctr,mainfold,:].T
                A=crit[:,maintrain]

                consensus_matrix = get_co_cluster_count(A)
                pac[current_set,ctr, k, mainfold] = get_pac(consensus_matrix)
                coph[current_set,ctr, k, mainfold] = coph_cor(consensus_matrix)

pkl_filename = (savedir + 'all_pac.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(pac, file)
pkl_filename = (savedir + 'all_coph.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(coph, file)

microF1 = np.full([len(sets),2,8,4,4],np.nan)
macroF1 = np.full([len(sets),2,8,4,4],np.nan)
for current_set in range(len(sets)):
    for ctr in range(2):
        for mainfold in range(4):
            for subfold in range(4):
                fold = (mainfold*4)+subfold

                if ctr==1:
                    pkl_filename = (savedir + sets[current_set] + '_mod_ctrl_' + str(fold))
                else:
                    pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
                print(pkl_filename)
                with open(pkl_filename, "rb") as file:
                    mod = pickle.load(file)

                microF1[current_set,ctr,:,mainfold,subfold] = np.nanmean(mod.micro_f1,axis=1)
                macroF1[current_set, ctr, :, mainfold, subfold] = np.nanmean(mod.macro_f1, axis=1)

pkl_filename = (savedir + 'all_microf1.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(microF1, file)
pkl_filename = (savedir + 'all_macrof1.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(macroF1, file)