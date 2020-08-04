import numpy as np
import csv

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

for current_set in range(12):
    data_path = '/Volumes/ELEMENTS/clustering_pilot/input_files/ALL_nomdd/ALL_nomdd_' + sets[current_set] + '_ctrl.csv'
    print(data_path)
    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        data = np.array(list(reader)).astype(float)

    X=data
    xperm = np.full(X.shape,np.nan)
    for c in range(X.shape[1]):
        feature = X[:,c]
        ss=np.linspace(0,X.shape[0]-1,X.shape[0])
        np.random.shuffle(ss)
        xperm[:,c]=feature[ss.astype(int)]

    ss='/Volumes/ELEMENTS/clustering_pilot/input_files/ALL_nomdd_null/ALL_nomdd_null_' + sets[current_set] + '_ctrl.csv'
    with open(ss,mode='w') as file:
        filewriter = csv.writer(file, delimiter=',')
        filewriter.writerows(xperm)
    file.close()





