import os
import pickle
import numpy as np
os.chdir('/u/ljollans/ML_in_python')
import loocv_assigmatcher_nov
from loocv_assigmatcher_nov import getloopcount,match2idx,getsubfoldbetas,loocv_assigmatcher,getSILCALBIC

sets=['Tc','Sc','TSc','Tc_tc','Sc_sc','TSc_tsc','Tct_s','Scs_s','Tct_Scs_s','Tct_tc_s','Scs_sc_s','Tct_Scs_tc_sc_s']

dir2='/u/ljollans/ML_in_python/allresidsjan2/full_covariance/'
savedir=(dir2 + 'FEB_')
 
ex1=getloopcount((dir2 + 'FEB'),0) # check where i have all files
[bestSIL,bestCAL,bestnclus_mf, bestnclus_sf,SIL,CAL,BIC]=getSILCALBIC(dir2 + 'FEB_',ex1,0)

import sys
s=int(sys.argv[1])
if s<len(sets):
    ctr=0
else:
    s=s-len(sets)
    ctr=1
print(s)
nclus2use=np.array([4,4,1,4,4,1,4,4,1,4,4,1])
nclus=nclus2use[s]
#nclus=bestnclus_mf[0,ctr,s].astype(int)
print(nclus)
allbetas=getsubfoldbetas(savedir,s,ctr, nclus,0)
if ctr==0:
    pkl_filename = (dir2 + 'FEB_' + sets[s] + 'BETA_AGGR' +  '.pkl')
else:
    pkl_filename = (dir2 + 'FEB_' + sets[s] + 'BETA_AGGR_ctrl'  + '.pkl')
with open(pkl_filename, 'wb') as file:
    pickle.dump(allbetas,file)  
print(pkl_filename)
 
               
