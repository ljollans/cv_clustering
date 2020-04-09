import csv
import numpy as np
import time
from clusmetwrapper import cluster
import pickle


pkl_filename = 'tst_set0.pkl'
with open(pkl_filename,"rb") as file:
    mod = pickle.load(file)

mod.hierarchical_consensus()