import pickle
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s", "Tct_Scs_tc_sc_s"]
n_cv_folds = 4
n = 398
n_k = 8

nclus_agglom=np.full([2,len(sets),n_cv_folds,n_cv_folds,n_k],np.nan)

#fig=plt.figure(figsize=[20,10])
for null in range(2):
    if null==1:
        input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/null/MDDnull/MDD__'
        modstr = '_mod_null_'
    else:
        input_filedir = '/Users/lee_jollans/Projects/clustering_pilot/FEB_PUT/FEB_'
        modstr = '_mod_ctrl_'

    for s in range(len(sets)):

        filestr = (input_filedir + sets[s] + modstr + str(0))
        with open(filestr, "rb") as f:
            mod = pickle.load(f)

        for mf in range(n_cv_folds):
            for k in range(n_k):

                X=np.empty((mod.data.shape[1],0),int)
                for sf in range(n_cv_folds):
                    fold=(mf*n_cv_folds)+sf
                    filestr=(input_filedir + sets[s] + modstr + str(fold))
                    with open(filestr, "rb") as f:
                        mod = pickle.load(f)

                    if k==0:
                        crit=np.nanmean(mod.allbetas[k],axis=1)
                        crit=np.array([crit,-crit]).T
                    else:
                        crit=np.nanmean(mod.allbetas[k],axis=2)

                    X = np.append(X,crit, axis=1)

                clustering = AgglomerativeClustering(compute_full_tree=True, distance_threshold=3, n_clusters=None,linkage='complete').fit(X.T)
                nclus_agglom[null,s,mf,sf,k]=clustering.n_clusters_

                clustering = AgglomerativeClustering(n_clusters=k+2, linkage='complete').fit(X.T)

                assig=np.full([n_cv_folds,k+2],np.nan)
                allbetas = np.full([X.shape[0], k + 2], np.nan)
                ctr=0
                for f in range(n_cv_folds):
                    for cc in range(k+2):
                        assig[f,cc]=clustering.labels_[ctr]
                        ctr+=1
                for cc in range(k+2):
                    allbetas[:,cc]=np.nanmean(X[:,np.where(clustering.labels_==cc)[0]], axis=1)

                with open((input_filedir + sets[s] + 'aggr_betas.pkl'), 'wb') as f:
                    pickle.dump([X,clustering,assig,allbetas],f)

with open((input_filedir + 'nclus_agglom.pkl'), 'wb') as f:
    pickle.dump(nclus_agglom, f)






#fig=plt.figure(figsize=[15,4])
#for n in range(3):
#    plt.subplot(3,2,1+(n*2)); plt.hist(nclus_agglom[0,n,:,:,:].flatten()); plt.title('actual')
#    plt.subplot(3,2,2+(n*2)); plt.hist(nclus_agglom[1,n,:,:,:].flatten()); plt.title('null')
#plt.show()