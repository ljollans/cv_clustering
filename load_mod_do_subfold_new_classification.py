import pickle

savedir = ('/Users/lee_jollans/Projects/clustering_pilot/ALL/wspecsamp_')

# presets
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

for current_set in range(len(sets)):

    for mainfold in range(4):
        for subfold in range(4):
            fold = (mainfold*4)+subfold

            pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
            print(pkl_filename)
            with open(pkl_filename, "rb") as file:
                mod = pickle.load(file)

            mod.cluster_ensembles()
            mod.cluster_ensembles_new_classification()

            with open(pkl_filename, "wb") as file:
                pickle.dump(mod,file)
