import numpy as np
import pickle
import csv
import cv_clustering
from cv_clustering.beta_aggregate import get_proba

input_files_dir = "/Users/lee_jollans/Projects/clustering_pilot/residfiles_all_210220/MDD__"
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_')
n=398
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

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


allargmax = np.full([n,12,8,4],np.nan)
allproba = np.full([n,12,8,4],np.nan)
for current_set in range(len(sets)):
    print(sets[current_set])
    data_path = input_files_dir + sets[current_set] + ".csv"
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)
    for k in range(8):
        for mainfold in range(4):
            maintrain = np.where(np.isfinite(cv_assignment[:, mainfold]))[0]
            ss = (savedir + sets[current_set] + '_aggregated_betas_k' + str(k) + '_' + str(mainfold) + '.csv')
            with open(ss, "r") as f:
                reader = csv.reader(f, delimiter=",")
                aggregated_betas = np.array(list(reader)).astype(float)

            Xtrain = data[maintrain,:]
            tmpX = np.append(np.ones(shape=[Xtrain.shape[0], 1]), Xtrain, axis=1)
            newY = tmpX.dot(aggregated_betas)
            argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
            tmp_trainproba, tmp_testproba = get_proba(Xtrain, argmaxY , aggregated_betas, data)

            argmaxY = np.array([np.where(tmp_testproba[i, :] == np.max(tmp_testproba[i, :]))[0][0] for i in range(tmp_testproba.shape[0])])
            valmaxY = np.array([np.max(tmp_testproba[i, :]) for i in range(tmp_testproba.shape[0])])

            allargmax[:,current_set,k,mainfold]=argmaxY
            allproba[:, current_set, k, mainfold] = valmaxY


pkl_filename = (savedir + 'all_labelsmain_sigmoid.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(allargmax, file)
pkl_filename = (savedir + 'all_probamain_sigmoid.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(allproba, file)
