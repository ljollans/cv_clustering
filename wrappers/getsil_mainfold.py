import numpy as np
import pickle
import csv
from sklearn.metrics import silhouette_samples, silhouette_score

ctr=0
input_files_dir = '/Users/lee_jollans/Projects/clustering_pilot/IXI2/act/IXI2_'
cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
savedir = ('/Users/lee_jollans/Projects/clustering_pilot/IXI2/act/IXI2_')

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

with open((cv_assignment_dir + "CVassigIXI.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

n=cv_assignment.shape[0]


if ctr==1:
    pkl_filename1 = (savedir + 'all_labelsmain_ctrl_sigmoid2.pkl')
else:
    pkl_filename1 = (savedir + 'all_labelsmain_sigmoid2.pkl')

print(pkl_filename1)
with open(pkl_filename1, "rb") as file:
    all_labelsmain = pickle.load(file)

all_n_sil = np.full([n,12,8,4],np.nan)
allsil = np.full([12,8,4],np.nan)
for current_set in range(len(sets)):

    print(sets[current_set])
    if ctr==0:
        data_path = input_files_dir + sets[current_set] + ".csv"
    else:
        data_path = input_files_dir + sets[current_set] + "_ctrl.csv"
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)

    for k in range(8):
        for mainfold in range(4):
            cluster_labels = all_labelsmain[:,current_set,k,mainfold]
            if len(np.unique(cluster_labels))>1:
                allsil[current_set, k, mainfold] = silhouette_score(data, cluster_labels)
                all_n_sil[:, current_set, k, mainfold] = silhouette_samples(data, cluster_labels)

if ctr==1:
    pkl_filename1 = (savedir + 'all_sil_ctrl_sigmoid2.pkl')
    pkl_filename2 = (savedir + 'all_n_sil_ctrl_sigmoid2.pkl')
else:
    pkl_filename1 = (savedir + 'all_sil_sigmoid2.pkl')
    pkl_filename2 = (savedir + 'all_n_sil_sigmoid2.pkl')

with open(pkl_filename1, "wb") as file:
    pickle.dump(allsil, file)

with open(pkl_filename2, "wb") as file:
    pickle.dump(all_n_sil, file)

