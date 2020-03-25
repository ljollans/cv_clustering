Cross-validated clustering pipeline
=======================================

Files
-----

* **cluster_wrapper_0.py** : sets data, cross-validation fold assignment, current main and subfold, covariance type to use for GMM, number of clusters to test
* **clusmetwrapper.py** : wrapper that points to looco_loop.py to compute cluster solutions per number of clusters and leave-one-out iteration and get the BIC, silhouette score, calinski-harabasz score, AUC, F1 and Beta values
* **looco_loop.py** : takes as input the data, covariance to use and number of clusters and computes a model with each observation left out once
* **clusmets.py** : takes as input the data, number of clusters to use and covariance type, computes a cluster solution, calculated BIC, silhouette score and calinski-harabasz score, and then passes the cluster assignments to multi-logr-bag.py to get beta weights and AUC and F1 score
* **multi_logr_bag.py** : takes as input the data, group assignments, number of groups, number of bagging iterations and number of cross-validation folds to run, and calculates a one-vs-all classifier for each group within 