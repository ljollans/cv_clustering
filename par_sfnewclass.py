import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering')
import pickle

from joblib import Parallel, delayed
import multiprocessing
import os
import sys
sys.path.append('/Users/lee_jollans/PycharmProjects/mdd_clustering/cv_clustering')


def sf_new_class(savedir,modstr='_mod_'):

    sets = ["Tc", "Sc", "TSc", "Tc_tc", "Sc_sc", "TSc_tsc", "Tct_s", "Scs_s", "Tct_Scs_s", "Tct_tc_s", "Scs_sc_s",
            "Tct_Scs_tc_sc_s"]
    num_cores = multiprocessing.cpu_count()
    for current_set in range(12):
        Parallel(n_jobs=num_cores)(delayed(do_sf_newcalc)((savedir + sets[current_set] + modstr + str(fold))) for fold in range(16))


def do_sf_newcalc(pkl_filename):
    print(pkl_filename)

    try:
        with open(pkl_filename, "rb") as file:
            mod = pickle.load(file)

        if hasattr(mod,'cluster_ensembles_labels'):
            print('already done')
            pass
        else:

            try:
                origdir = os.getcwd()
                try:
                    os.mkdir(pkl_filename + 'tmp')
                except:
                    pass
                os.chdir(pkl_filename + 'tmp')

                success = 0;             ctr = 0
                while success == 0:
                    try:
                        mod.cluster_ensembles()
                        success = 1
                    except:
                        ctr += 1
                        print('fail ' + str(ctr))

                os.remove('Cluster_Ensembles.h5')
                os.chdir(origdir)
                os.rmdir(pkl_filename + 'tmp')
            except:
                pass

        if hasattr(mod,'highest_prob'):
            pass
        else:
            mod.cluster_ensembles_new_classification()
            with open(pkl_filename, "wb") as file:
                pickle.dump(mod, file)
    except:
        pass
