import csv
import numpy as np
import time
from clusmetwrapper import cluster
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from utils import get_gradient_change

sets = [
    "Tc",
    "Sc",
    "TSc",
    "Tc_tc",
    "Sc_sc",
    "TSc_tsc",
    "Tct_s",
    "Scs_s",
    "Tct_Scs_s",
    "Tct_tc_s",
    "Scs_sc_s",
    "Tct_Scs_tc_sc_s",
]

pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_rand.pkl'
with open(pkl_filename,"rb") as file:
    mdd_rand = pickle.load(file)
pkl_filename = '/Users/lee_jollans/Projects/clustering_pilot/all_pac.pkl'
with open(pkl_filename,"rb") as file:
    mdd_pac = pickle.load(file)


optg = np.full([12,2,4,4],np.nan)
usek=np.full([12,2,4],np.nan)
for s in range(12):
    for ctr in range(2):
        for mf in range(4):
            ttmp=np.full([4,7],np.nan)
            for sf in range(4):
                crit=mdd_pac[:,s,ctr,mf,sf]
                ttmp[sf,:] = get_gradient_change(crit)
                optg[s, ctr, mf, sf] = np.where(ttmp[sf,:] == np.nanmin(ttmp[sf,:]))[0]
            u=np.zeros(len(crit))
            for i in range(len(crit)):
                u[i]=len(np.where(optg[s,ctr,mf,:]==i)[0])
            if len(np.where(u>=2)[0])!=1:
                crit=np.nanmean(ttmp[:,:],axis=0)
                usek[s,ctr,mf]=np.where(crit==np.nanmin(crit))[0]+1
            else:
                usek[s,ctr,mf]=np.where(u>=2)[0]+1


fig=plt.figure()
for s in range(12):
    plt.subplot(4,3,s+1)
    plt.title(sets[s])
    for mf in range(4):
        plt.plot((np.nanmean(mdd_pac[:,s,0,mf,:],axis=1)))
    plt.legend(['0','1','2','3'])
plt.show()

# plomk them all into a pandas dataframe
df = pd.DataFrame({}, columns=['pac', 'rand', 'set','ctr','mf','sf','k'])
for s in range(12):
    for ctr in range(2):
        for mf in range(4):
            for sf in range(4):
                for k in range(8):
                    tmp_df = pd.DataFrame(
                        {'pac': [mdd_pac[k,s,ctr,mf,sf]],
                         'rand': [mdd_rand[k,s,ctr,mf,sf]],
                         'set': sets[s],
                         'ctr': ctr,
                         'mf': mf,
                         'sf': sf,
                         'k': int(k)+2},
                        columns=['pac', 'rand', 'set','ctr','mf','sf','k'])
                    df = df.append(tmp_df)

fig=plt.figure()
for s in range(12):
    plt.subplot(4,3,s+1)
    sns.lineplot(y='pac',x='k',hue='ctr',data=df[df['set']==sets[s]])