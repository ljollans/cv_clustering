import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from looco_loop import looco_loop
from multi_logr_bag import multi_logr_bagr
from multi_logr_bag_utils import bag_log
from utils import colorscatter
import scipy

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target
ss = np.linspace(0, len(Y) - 1, len(Y))
np.random.shuffle(ss)
X = X[ss.astype(int), :]
Y = Y[ss.astype(int)]

(
    auc,
    f1,
    betas2use,
    overall_prediction_continuous,
    overall_prediction_discrete,
    auc_partial,
    f1_partial,
    betas
) = multi_logr_bagr(10, np.append(np.expand_dims(Y, axis=1), X, axis=1), 3, 3, 0)

print(auc)
print()
print(auc_partial)
