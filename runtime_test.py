from clusmetwrapper import extract_vals
import matplotlib.pyplot as plt
import seaborn as sns

sets = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s',
         'Scs_sc_s', 'Tct_Scs_tc_sc_s']

filedir='/Users/lee_jollans/Projects/clustering_pilot//FEB_PUT//FEB_'
all_coph, df = extract_vals(filedir, sets, 'COPH', 8, 4, 1, 0)


fig=plt.figure()
for s in range(12):
    plt.subplot(4,3,s+1)
    sns.lineplot(y='COPH',x='k',hue='ctr',data=df[df['set']==sets[s]])
    plt.title(sets[s])
plt.show()