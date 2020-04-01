from loocv_assigmatcher_nov import getSILCALBIC, getloopcount, plot_bic_violin, get_clusassignments_from_LOOCV
from utils import (
    select_trainset,
    n_clus_retrieval_chk, get_co_cluster_count, get_maxmatches, percent_overlap_vectors, get_final_assignment)
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

set = 1
subfold = 0
mainfold = 0

A2=get_clusassignments_from_LOOCV(set,mainfold,subfold)
A=A2[:,4,:]
co_cluster_count=get_co_cluster_count(A)

maxmatches, findmax = get_maxmatches(A, co_cluster_count, .8)

print(maxmatches.shape)

final_assignment = get_final_assignment(maxmatches, 25)
print(len(final_assignment))