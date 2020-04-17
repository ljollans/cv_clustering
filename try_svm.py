import pickle
import csv
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from utils import select_trainset
from clusmetwrapper import kcluster

pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/ALL/ALL_Tc_mod_1'
with open(pkl_filename, "rb") as file:
    mod = pickle.load(file)

#mod.cluster_ensembles()
#with open(pkl_filename, "wb") as file:
#    pickle.dump(mod,file)

mod.cluster_ensembles_new_classification()


k = 4

clf = LogisticRegression(random_state=0, multi_class='ovr')
clf.intercept_ = np.nanmean(mod.allitcpt[k], axis=1)
clf.coef_ = np.nanmean(mod.allbetas[k], axis=2)

clf_isotonic = CalibratedClassifierCV(clf, cv=5, method='isotonic').fit(mod.data[mod.train_index_sub, :],
                                                                        mod.cluster_ensembles_labels[:, k])
clf_isotonic.predict_proba(mod.data[mod.test_index_sub, :])

a=clf_isotonic.predict_proba(mod.data[mod.test_index_sub, :])
b=clf_isotonic.predict_proba(mod.data[mod.train_index_sub, :])

byproba = [b[i,mod.cluster_ensembles_labels[i,k].astype(int)] for i in range(len(b))]
bytopa = [np.max(b[i,:]) for i in range(len(b))]
