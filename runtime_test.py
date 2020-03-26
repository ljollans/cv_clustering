import numpy as np
import matplotlib.pyplot as plt
from utils import many_co_cluster_match
import time
import pickle

n_groups=6

pkl_filename='/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT/FEB_Svcs_svc_sallcluslabels_fold7.pkl'
with open(pkl_filename, 'rb') as file:
    A = pickle.load(file)
A=np.squeeze(A[:,n_groups-2,:])

seconds = time.time()
co_cluster_count, maxmatches, overlap = many_co_cluster_match(A, n_groups)
print("Seconds since epoch =", time.time()-seconds)
#plt.hist(n_overlap.flatten()); plt.show()
plt.imshow(overlap); plt.colorbar(); plt.show()


#print(n_overlap.shape)
#plt.imshow(maxmatches); plt.colorbar(); plt.show()


#find_highest_overlap=np.where(n_overlap==np.max(n_overlap))