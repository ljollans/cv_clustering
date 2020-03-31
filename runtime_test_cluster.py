import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
from looco_loop import loocv_loop
from utils import colorscatter
import scipy

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
Y = iris.target

ss=np.linspace(0,len(Y)-1,len(Y))
np.random.shuffle(ss)
X=X[ss.astype(int),:]
Y=Y[ss.astype(int)]

(
    looco_all_clus_labels,
    looco_bic,
    looco_sil,
    looco_cal,
    looco_auc,
    looco_f1,
    looco_betas,
) = loocv_loop(X, "full", 1)

fig = plt.figure(figsize=[15, 5])
colorscatter(X, Y, np.ones(shape=X.shape[0]),plt.subplot(1, 2, 1))
all_assigs_mode=[scipy.stats.mode(looco_all_clus_labels[:,p])[0][0].astype(int)+1 for p in range(looco_all_clus_labels.shape[1])]
print(all_assigs_mode)
colorscatter(X, all_assigs_mode, np.ones(shape=X.shape[0]),plt.subplot(1, 2, 2))
plt.show()