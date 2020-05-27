import numpy as np
import pickle

from cv_clustering.beta_aggregate import aggregate, get_proba

input_files_dir = '/Users/lee_jollans/Projects/clustering_pilot/IXI2/act/IXI2_'
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/IXI2/null/IXI2_')

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
import csv

with open((cv_assignment_dir + "CVassigIXI.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

n=cv_assignment.shape[0]

allargmax = np.full([n,12,8,4],np.nan)
allproba = np.full([n,12,8,4],np.nan)

for current_set in range(len(sets)):
    data_path = input_files_dir + sets[current_set] + ".csv"
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)

    for k in range(8):
        trainproba = []
        testproba = []

        for mainfold in range(4):
            maintrain = np.where(np.isfinite(cv_assignment[:, mainfold]))[0]
            allbetas=[]
            allitcps = []
            labels = []

            for subfold in range(4):
                fold = (mainfold*4)+subfold

                pkl_filename = (savedir + sets[current_set] + '_mod_null_' + str(fold))
                print(pkl_filename)
                with open(pkl_filename, "rb") as file:
                    mod = pickle.load(file)

                labels.append(mod.cluster_ensembles_labels[:,k])
                i=np.nanmean(mod.allitcpt[k], axis=1)
                if k>0:
                    b=np.nanmean(mod.allbetas[k],axis=2)
                else:
                    b = np.nanmean(mod.allbetas[k], axis=1)
                    b = np.expand_dims(b, axis=1)
                    b = np.append(b, -b, axis=1)
                print(b.shape)
                allbetas.append(np.append(np.expand_dims(i,axis=1).T,b,axis=0))

            aggregated_betas, all_weighted_betas = aggregate(allbetas)
            ss=(savedir + sets[current_set] + '_aggregated_betas_k2' + str(k) + '_' + str(mainfold) + '.csv')
            with open(ss,mode='w') as file:
                filewriter = csv.writer(file, delimiter=',')
                filewriter.writerows(aggregated_betas)
            file.close()

            Xtrain = data[maintrain, :]
            tmpX = np.append(np.ones(shape=[Xtrain.shape[0], 1]), Xtrain, axis=1)
            newY = tmpX.dot(aggregated_betas)
            argmaxY = np.array([np.where(newY[i, :] == np.max(newY[i, :]))[0][0] for i in range(newY.shape[0])])
            tmp_trainproba, tmp_testproba = get_proba(Xtrain, argmaxY, aggregated_betas, data)

            argmaxY = np.array([np.where(tmp_testproba[i, :] == np.max(tmp_testproba[i, :]))[0][0] for i in
                                range(tmp_testproba.shape[0])])
            valmaxY = np.array([np.max(tmp_testproba[i, :]) for i in range(tmp_testproba.shape[0])])

            allargmax[:, current_set, k, mainfold] = argmaxY
            allproba[:, current_set, k, mainfold] = valmaxY

pkl_filename = (savedir + 'all_labelsmain_sigmoid2.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(allargmax, file)
pkl_filename = (savedir + 'all_probamain_sigmoid2.pkl')
with open(pkl_filename, "wb") as file:
    pickle.dump(allproba, file)

