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


#plt.imshow(A2); plt.colorbar(); plt.show()
co_cluster_count = get_co_cluster_count(A2)
plt.imshow(co_cluster_count); plt.colorbar(); plt.show()