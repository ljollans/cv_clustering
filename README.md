Cross-validated clustering pipeline - step 1: clustering
=======================================

Files
-----

* **cluster_wrapper_0.py** : sets data, cross-validation fold assignment, current main and subfold, covariance type to use for GMM, number of clusters to test
* **clusmetwrapper.py** : wrapper that points to loocv_loop.py to compute cluster solutions per number of clusters and leave-one-out iteration and get the BIC, silhouette score, calinski-harabasz score, AUC, F1 and Beta values
* **loocv_loop.py** : takes as input the data, covariance to use and number of clusters and computes a model with each observation left out once
* **clusmets.py** : takes as input the data, number of clusters to use and covariance type, computes a cluster solution, calculated BIC, silhouette score and calinski-harabasz score, and then passes the cluster assignments to multi-logr-bag.py to get beta weights and AUC and F1 score
* **multi_logr_bag.py** : takes as input the data, group assignments, number of groups, number of bagging iterations and number of cross-validation folds to run, and calculates a one-vs-all classifier for each group within a bagged cross-validation framework. returned beta weights are averages across iterations

Input data
----------

User input to clusmetwrapper.py:
* design['X']: The data
* design['n_ks']: The range for number of clusters to test (i.e. '8' would result in between 2 and 10 clusters being tested)
* design['cv_assignment']: The cross-validation fold assignment in the shape of a [N x n_cv_folds] matrix, where each column represents a cross-validation fold. In each column numbers from 0 to n_cv_folds indicate the nested CV fold within the outer training set with NaN indicating the outer test set.
* design['mainfold']: The outer cross-validation fold to use for the analysis iteration
* design['subfold']: The nested cross-validation fold to use for the analysis iteration
* design['covariance']: The covariance type to use for GMM (e.g. full or spherical)

Output 
-------------------

Output from clusmetwrapper.py:
* all_clus_labels ([N x n_ks x N1] matrix): cluster assignments for all observations (N1) by LOOCV iteration and number of clusters
* bic ([N * n_ks] matrix): BIC values for each LOOCV iteration and number of clusters
* sil ([N * n_ks] matrix): silhouette scores for each LOOCV iteration and number of clusters
* cal ([N * n_ks] matrix): calinski-harabasz scores for each LOOCV iteration and number of clusters
* auc ([N * n_ks] matrix): AUC values for each LOOCV iteration and number of clusters
* f1 ([N * n_ks] matrix): F1 scores for each LOOCV iteration and number of clusters
* betas ([N * n_ks * n_ks1 * n_features] matrix): BETA weights for each cluster (n_ks1) by LOOCV iteration and number of clusters

Files saved by cluster_wrapper_0.py:
* ["BIC", "SIL", "CAL", "all_clus_labels", "AUC", "F1", "BETAS"]
* saved under the specified save directory and with the save string prefix with the used dataset and nested cv fold (e.g. a number form 0-15 for 4x4 cross-validation)