import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def identify_set_and_fold(current_proc, n_cv_folds):
    folds_per_run = n_cv_folds * n_cv_folds
    current_set = np.floor_divide(current_proc, folds_per_run)
    current_set_count_start = current_set * folds_per_run
    current_fold = current_proc - current_set_count_start
    return current_set, current_fold

def colorscatter(X_e,Y,d,ax):
    try:
        groups=(set(Y[~np.isnan(Y)]))
    except:
        groups=np.unique(Y)
    colors = matplotlib.cm.tab10(np.linspace(0, 1, 10))
    ctr=-1
    for g in groups:
        ctr+=1
        findme=np.where(Y==g.astype(int))[0]
        cc=np.expand_dims(colors[ctr],axis=0)
        ax.scatter(X_e[findme,0],X_e[findme,1],c=cc, s=d[findme]*5)
    plt.legend(groups)
