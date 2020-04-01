from loocv_assigmatcher_nov import getSILCALBIC, getloopcount, plot_bic_violin, k_workup_mainfold
from utils import (
    select_trainset,
    n_clus_retrieval_chk)
import pickle
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    cv_assignment = np.array(list(reader)).astype(float)

mainfold=0

k_workup_mainfold(mainfold,7)