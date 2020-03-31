import numpy as np
from utils import (
    contingency_matrix,
    get_co_cluster_count,
    get_maxmatches,
    check_equality,
    percent_overlap_vectors,
    select_testset,
    select_trainset
)
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import ecdf




n_groups = 6
discretion_level = 0.2

cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)



pkl_filename = "/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_svc_sallcluslabels_fold1.pkl"
with open(pkl_filename, "rb") as file:
    A = pickle.load(file)
A = np.squeeze(A[:, n_groups - 2, :])
trainset=select_trainset(cv_assignment,0,1)
A1=A[trainset,:]
A2=A1[:,trainset]



co_cluster_count = get_co_cluster_count(A2)
co_cluster_count=(co_cluster_count)/(A2.shape[1]-2)
con_mat=co_cluster_count

x1 = 0.1
x2 = 0.9
p = con_mat[np.tril_indices(con_mat.shape[0])]  .flatten()

xs,ys = ecdf ( p )
select_vals=np.where(np.logical_and(xs>=x1, xs<=x2))
x1_x2_range = ys[select_vals[0]]
x1_val = x1_x2_range[ 0 ]
x2_val = x1_x2_range[ -1 ]
PAC1 = x2_val - x1_val
print(PAC1)

pac_temp = ecdf ( p )
pac_temp = pd.DataFrame ( {'index': pac_temp[ 0 ], 'cdf': pac_temp[ 1 ]} )
pac_temp = pac_temp[ pac_temp[ 'index' ].isin ( [ x1, x2 ] ) ]
x1_val = pac_temp[ 'cdf' ].iloc[ 0 ]
x2_val = pac_temp[ 'cdf' ].iloc[ -1 ]
PAC2 = x2_val - x1_val
print(PAC2)
