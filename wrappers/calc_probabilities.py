import csv
import numpy as np
import pickle
from beta_aggregate import get_proba

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

# all_subfold_betas is a list of length n_cv
# each entry has the shape nfeatures x nclusters
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

testprobabilities = np.full([8, len(sets), 2, 4], np.nan)

for current_set in range(len(sets)):
    for ctr in range(2):

        # data import
        if ctr == 1:
            data_path = input_files_dir + "MDD__" + sets[current_set - len(sets)] + "_ctrl.csv"
        else:
            data_path = input_files_dir + "MDD__" + sets[current_set] + ".csv"
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            data = np.array(list(reader)).astype(float)

        for kloop in range(7):
            k = kloop + 1

            trainproba = []
            testproba = []

            for mainfold in range(4):
                maintrain = np.where(np.isfinite(cv_assignment[:, 0]))[0]
                Xtrain = data[maintrain, :]
                maintest = np.where(np.isnan(cv_assignment[:, 0]))[0]
                Xtest = data[maintest, :]

                with open((savedir + sets[current_set] + '_aggregated_betas_k' + str(k) + '_' + str(mainfold) + '.csv'),
                          mode='r') as file:
                    filereader = csv.reader(file, delimiter=',')
                    aggregated_betas = np.array(list(filereader)).astype(float)
                file.close()

                tmpX = np.append(np.ones(shape=[data.shape[0], 1]), data, axis=1)
                newY = tmpX.dot(aggregated_betas)
                argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
                trainlabels = [argmaxY[i] for i in maintrain]

                tmp_trainproba, tmp_testproba = get_proba(Xtrain, trainlabels, aggregated_betas, Xtest)
                trainproba.append(tmp_trainproba)
                testproba.append(tmp_testproba)

                crit = testproba[mainfold]
                a = np.array([np.max(crit[i, :]) for i in range(crit.shape[0])])
                testprobabilities[k, current_set, ctr, mainfold] = np.nanmean(a)

            if ctr==1:
                pkl_filename1 = (savedir + sets[current_set] + '_trainprob_k_ctrl' + str(k) + '.pkl')
                pkl_filename2 = (savedir + sets[current_set] + '_testprob_k_ctrl' + str(k) + '.pkl')
            else:
                pkl_filename1 = (savedir + sets[current_set] + '_trainprob_k' + str(k) + '.pkl')
                pkl_filename2 = (savedir + sets[current_set] + '_testprob_k' + str(k) + '.pkl')

            with open(pkl_filename1, "wb") as file:
                pickle.dump(trainproba, file)
            with open(pkl_filename2, "wb") as file:
                pickle.dump(testproba, file)

pkl_filename = (savedir + 'testprobabilities.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(testprobabilities, file)