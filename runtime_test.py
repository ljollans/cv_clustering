from clusmetwrapper import extract_vals
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

sets = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
        'Scs_sc_s', 'Tct_Scs_tc_sc_s']

filedir = '/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT//FEB_'
s = 0
pkl_filename = filedir + '_BetaAggr.pkl'
with open(pkl_filename, "rb") as file:
    BetaAggr = pickle.load(file)

mfs = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])

mf = 0
k = 4
all_subfold_betas=[BetaAggr[np.where(mfs==mf)[0][i]][k] for i in range(4)]