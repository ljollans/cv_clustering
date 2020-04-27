import pickle

savedir = ('/Users/lee_jollans/Projects/clustering_pilot/IXI/IXI_')

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

for current_set in [11]:

    for mainfold in [3]:
        for subfold in [3]:
            fold = (mainfold*4)+subfold

            pkl_filename = (savedir + sets[current_set] + '_mod_' + str(fold))
            print(pkl_filename)
            with open(pkl_filename, "rb") as file:
                mod = pickle.load(file)

            mod.cluster_ensembles()
            mod.cluster_ensembles_new_classification()

            with open(pkl_filename, "wb") as file:
                pickle.dump(mod,file)


