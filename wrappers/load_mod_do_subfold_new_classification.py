import pickle
from cv_clustering import clusmetwrapper
from joblib import Parallel, delayed
import multiprocessing
import os

def sf_new_class(savedir,modstr='_mod_'):

    sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s",
            "Tct_Scs_tc_sc_s"]
    num_cores = multiprocessing.cpu_count()
    for current_set in range(12):
        Parallel(n_jobs=4)(delayed(do_sf_newcalc)((savedir + sets[current_set] + modstr + str(fold))) for fold in range(16))


def do_sf_newcalc(pkl_filename):
    with open(pkl_filename, "rb") as file:
        mod = pickle.load(file)
    print(pkl_filename)
    if hasattr(mod,'cluster_ensembles_labels'):
        print('already done')
        pass
    else:

        origdir = os.getcwd()
        try:
            os.mkdir(pkl_filename + 'tmp')
        except:
            pass
        os.chdir(pkl_filename + 'tmp')
        mod.cluster_ensembles()
        os.remove('Cluster_Ensembles.h5')
        os.chdir(origdir)
        os.rmdir(pkl_filename + 'tmp')

    if hasattr(mod, 'highest_prob'):
        pass
    else:
        mod.cluster_ensembles_new_classification()
        with open(pkl_filename, "wb") as file:
            pickle.dump(mod, file)