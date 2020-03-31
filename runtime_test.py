import numpy as np
from utils import (
    contingency_matrix,
    get_co_cluster_count,
    get_maxmatches,
    check_equality,
    percent_overlap_vectors,
    select_testset,
    select_trainset,
    n_clus_retrieval_chk)
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stats
from utils import ecdf


cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

mainfold=0

fig=plt.figure()
ctr=0
for subfold in range(4):

    pkl_filename = ('/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_sallcluslabels_fold' + str(subfold) + '.pkl')
    with open(pkl_filename, "rb") as file:
        Aorig = pickle.load(file)


    trainset=select_trainset(cv_assignment,mainfold,subfold)
    A1=Aorig[trainset,:,:]
    A2=A1[:,:,trainset]
    cr, pac = n_clus_retrieval_chk ( A2 )

    ctr+=1
    plt.subplot(4,2,ctr)
    plt.plot(cr)
    ctr += 1
    plt.subplot ( 4, 2, ctr)
    plt.plot(pac)
plt.show()

