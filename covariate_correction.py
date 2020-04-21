import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import csv

def covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct):
    # use same data to correct and build
    Xcorrected = np.full([data2Correct.shape[0], data2Correct.shape[1]], np.nan)
    for i in range(0, data2Correct.shape[1]):
        modelcase = sm.OLS(data2Correct[:, i], vars2Correct).fit()
        Xcorrected[:, i] = data2Correct[:, i] - modelcase.predict(vars2Correct)
    Xcorrected_zscoredsame = scaler.fit_transform(Xcorrected)

    # use other data to correct and build
    Xcorrected = np.full([data2Correct.shape[0], data2Correct.shape[1]], np.nan)
    for i in range(0, data2Build.shape[1]):
        modelcase = sm.OLS(data2Build[:, i], vars2Build).fit()
        Xcorrected[:, i] = data2Correct[:, i] - modelcase.predict(vars2Correct)
    Xcorrected_zscoredother = scaler.fit_transform(Xcorrected)

    return Xcorrected_zscoredsame, Xcorrected_zscoredother


def corrections(Xcase, Xctrl, Xcase_cov, Xctrl_cov):
    # all datasets are ordered the same
    surf_v = np.linspace(0, 67, 68).astype(int)
    thick_v = np.linspace(68, 67 + 68, 68).astype(int)
    subc_v = np.append(np.linspace(137, 143, 6), np.linspace(145, 151, 6)).astype(int)
    cov_v = [136, 144]
    cov_s = [154, 155]
    cov_t = [152, 153]
    cov_c = [156, 157, 158, 159]

    # zscore
    Xctrlz = scaler.fit_transform(Xctrl)
    Xcasez = scaler.fit_transform(Xcase)
    Xctrl_covz = scaler.fit_transform(Xctrl_cov)
    Xcase_covz = scaler.fit_transform(Xcase_cov)

    # correct the case data for ventricle volumes and the standard covariate set
    data2Correct = Xcasez
    data2Build = Xctrlz
    vars2Build = np.append(Xctrl_covz, Xctrlz[:, cov_v], axis=1)
    vars2Correct = np.append(Xcase_covz, Xcasez[:, cov_v], axis=1)
    [Xcase_vcz, Xcase_ctrl_vcz] = covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct)

    # correct the case data for ventricle volumes, surface area summary measures and the standard covariate set
    data2Correct = Xcasez
    data2Build = Xctrlz
    vars2Build = np.append(np.append(Xctrl_covz, Xctrlz[:, cov_v], axis=1), Xctrlz[:, cov_s], axis=1)
    vars2Correct = np.append(np.append(Xcase_covz, Xcasez[:, cov_v], axis=1), Xcasez[:, cov_s], axis=1)
    [Xcase_vcsz, Xcase_ctrl_vcsz] = covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct)

    # correct the case data for ventricle volumes, thickness summary measures and the standard covariate set
    data2Correct = Xcasez
    data2Build = Xctrlz
    vars2Build = np.append(np.append(Xctrl_covz, Xctrlz[:, cov_v], axis=1), Xctrlz[:, cov_t], axis=1)
    vars2Correct = np.append(np.append(Xcase_covz, Xcasez[:, cov_v], axis=1), Xcasez[:, cov_t], axis=1)
    [Xcase_vctz, Xcase_ctrl_vctz] = covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct)

    return Xctrlz, Xcasez, Xctrl_covz, Xcase_covz, Xcase_vcz, Xcase_ctrl_vcz, Xcase_vcsz, Xcase_ctrl_vcsz, Xcase_vctz, Xcase_ctrl_vctz


def corrections_noventricles(Xcase, Xctrl, Xcase_cov, Xctrl_cov):
    # all datasets are ordered the same
    surf_v = np.linspace(0, 67, 68).astype(int)
    thick_v = np.linspace(68, 67 + 68, 68).astype(int)
    subc_v = np.append(np.linspace(137, 143, 6), np.linspace(145, 151, 6)).astype(int)
    cov_v = [136, 144]
    cov_s = [154, 155]
    cov_t = [152, 153]
    cov_c = [156, 157, 158, 159]

    # zscore
    Xctrlz = scaler.fit_transform(Xctrl)
    Xcasez = scaler.fit_transform(Xcase)
    Xctrl_covz = scaler.fit_transform(Xctrl_cov)
    Xcase_covz = scaler.fit_transform(Xcase_cov)

    # correct the case data for  the standard covariate set
    data2Correct = Xcasez
    data2Build = Xctrlz
    vars2Build = Xctrl_covz
    vars2Correct = Xcase_covz
    [Xcase_cz, Xcase_ctrl_cz] = covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct)

    # correct the case data for  surface area summary measures and the standard covariate set
    data2Correct = Xcasez
    data2Build = Xctrlz
    vars2Build = np.append(Xctrl_covz, Xctrlz[:, cov_s], axis=1)
    vars2Correct = np.append(Xcase_covz, Xcasez[:, cov_s], axis=1)
    [Xcase_csz, Xcase_ctrl_csz] = covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct)

    # correct the case data for thickness summary measures and the standard covariate set
    data2Correct = Xcasez
    data2Build = Xctrlz
    vars2Build = np.append(Xctrl_covz, Xctrlz[:, cov_t], axis=1)
    vars2Correct = np.append(Xcase_covz, Xcasez[:, cov_t], axis=1)
    [Xcase_ctz, Xcase_ctrl_ctz] = covariateCorrection(data2Correct, data2Build, vars2Build, vars2Correct)

    return Xctrlz, Xcasez, Xctrl_covz, Xcase_covz, Xcase_cz, Xcase_ctrl_cz, Xcase_csz, Xcase_ctrl_csz, Xcase_ctz, Xcase_ctrl_ctz


def makeVariableSets(Xcase_covz, Xcase_vcz, Xcase_vcsz, Xcase_vctz):
    # all datasets are ordered the same
    surf_v = np.linspace(0, 67, 68).astype(int)
    thick_v = np.linspace(68, 67 + 68, 68).astype(int)
    subc_v = np.append(np.linspace(137, 143, 7), np.linspace(145, 151, 7)).astype(int)
    cov_v = [136, 144]
    cov_s = [154, 155]
    cov_t = [152, 153]
    cov_c = [156, 157, 158, 159]

    # corrected for ventricle volumes and the standard covariate set
    # no covariates
    Tvc = np.append(Xcase_vcz[:, thick_v], Xcase_vcz[:, subc_v], axis=1)  # thickness and subcortical
    Svc = np.append(Xcase_vcz[:, surf_v], Xcase_vcz[:, subc_v], axis=1)  # SA and subcortical
    TSvc = Xcase_vcz[:, np.concatenate([thick_v, surf_v, subc_v], axis=0)]  # thickness, SA, and subcortical
    # with covariates
    Tvc_tvc = np.append(Tvc, Xcase_vcz[:, cov_t], axis=1)  # thickness and subcortical
    Svc_svc = np.append(Svc, Xcase_vcz[:, cov_s], axis=1)  # SA and subcortical
    TSvc_tsvc = Xcase_vcz[:,
                np.concatenate([thick_v, surf_v, subc_v, cov_t, cov_s], axis=0)]  # thickness, SA, and subcortical

    # corrected thickness for thickness summary measures, ventricle volumes and the standard covariate set
    # corrected SA for SA summary measures, ventricle volumes and the standard covariate set
    # corrected subcortical for ventricle volumes and the standard covariate set
    # no covariates
    Tvct_s = np.append(Xcase_vctz[:, thick_v], Xcase_vcz[:, subc_v], axis=1)  # thickness and subcortical
    Svcs_s = np.append(Xcase_vcsz[:, surf_v], Xcase_vcz[:, subc_v], axis=1)  # SA and subcortical
    Tvct_Svcs_s = np.append(np.append(Xcase_vctz[:, thick_v], Xcase_vcsz[:, surf_v], axis=1), Xcase_vcz[:, subc_v],
                            axis=1)  # thickness, SA, and subcortical
    # with covariates
    Tvct_tvc_s = np.append(Tvct_s, Xcase_vcz[:, cov_t], axis=1)  # thickness and subcortical
    Svcs_svc_s = np.append(Svcs_s, Xcase_vcz[:, cov_s], axis=1)  # SA and subcortical
    Tvct_Svcs_tvc_svc_s = np.append(np.append(Tvct_Svcs_s, Xcase_vcz[:, cov_t], axis=1), Xcase_vcz[:, cov_s],
                                    axis=1)  # thickness, SA, and subcortical

    return Tvc, Svc, TSvc, Tvc_tvc, Svc_svc, TSvc_tsvc, Tvct_s, Svcs_s, Tvct_Svcs_s, Tvct_tvc_s, Svcs_svc_s, Tvct_Svcs_tvc_svc_s


# REMOVED
## corrected thickness and subcortical for thickness summary measures, ventricle volumes and the standard covariate set
## corrected SA and subcortical for SA summary measures, ventricle volumes and the standard covariate set
## no covariates
# Tvct=np.append(Xcase_vctz[:,thick_v],Xcase_vctz[:,subc_v],axis=1) # thickness and subcortical
# Svcs=np.append(Xcase_vcsz[:,surf_v],Xcase_vcsz[:,subc_v],axis=1) # SA and subcortical
# Tvct_Svcs=np.append(Tvct,Svcs,axis=1) # thickness, SA, and subcortical
## with covariates
# Tvct_tvc=np.append(Tvct, Xcase_vcz[:,cov_t],axis=1) # thickness and subcortical
# Svcs_svc=np.append(Svcs, Xcase_vcz[:,cov_s],axis=1) # SA and subcortical
# Tvct_Svcs_tvc_svc=np.append(Tvct_tvc,Svcs_svc,axis=1) # thickness, SA, and subcortical

def makeVariableSets_noventricles(Xcase_covz, Xcase_cz, Xcase_csz, Xcase_ctz):
    # all datasets are ordered the same
    surf_v = np.linspace(0, 67, 68).astype(int)
    thick_v = np.linspace(68, 67 + 68, 68).astype(int)
    subc_v = np.append(np.linspace(137, 143, 7), np.linspace(145, 151, 7)).astype(int)
    cov_v = [136, 144]
    cov_s = [154, 155]
    cov_t = [152, 153]
    cov_c = [156, 157, 158, 159]

    # corrected for  the standard covariate set
    # no covariates
    Tc = np.append(Xcase_cz[:, thick_v], Xcase_cz[:, subc_v], axis=1)  # thickness and subcortical
    Sc = np.append(Xcase_cz[:, surf_v], Xcase_cz[:, subc_v], axis=1)  # SA and subcortical
    TSc = Xcase_cz[:, np.concatenate([thick_v, surf_v, subc_v], axis=0)]  # thickness, SA, and subcortical
    # with covariates
    Tc_tc = np.append(Tc, Xcase_cz[:, cov_t], axis=1)  # thickness and subcortical
    Sc_sc = np.append(Sc, Xcase_cz[:, cov_s], axis=1)  # SA and subcortical
    TSc_tsc = Xcase_cz[:,
              np.concatenate([thick_v, surf_v, subc_v, cov_t, cov_s], axis=0)]  # thickness, SA, and subcortical

    # corrected thickness for thickness summary measures, and the standard covariate set
    # corrected SA for SA summary measures,  and the standard covariate set
    # corrected subcortical for  and the standard covariate set
    # no covariates
    Tct_s = np.append(Xcase_ctz[:, thick_v], Xcase_cz[:, subc_v], axis=1)  # thickness and subcortical
    Scs_s = np.append(Xcase_csz[:, surf_v], Xcase_cz[:, subc_v], axis=1)  # SA and subcortical
    Tct_Scs_s = np.append(np.append(Xcase_ctz[:, thick_v], Xcase_csz[:, surf_v], axis=1), Xcase_cz[:, subc_v],
                          axis=1)  # thickness, SA, and subcortical
    # with covariates
    Tct_tc_s = np.append(Tct_s, Xcase_cz[:, cov_t], axis=1)  # thickness and subcortical
    Scs_sc_s = np.append(Scs_s, Xcase_cz[:, cov_s], axis=1)  # SA and subcortical
    Tct_Scs_tc_sc_s = np.append(np.append(Tct_Scs_s, Xcase_cz[:, cov_t], axis=1), Xcase_cz[:, cov_s],
                                axis=1)  # thickness, SA, and subcortical

    return Tc, Sc, TSc, Tc_tc, Sc_sc, TSc_tsc, Tct_s, Scs_s, Tct_Scs_s, Tct_tc_s, Scs_sc_s, Tct_Scs_tc_sc_s


def makeAndSave(Xcase, Xctrl, Xcase_cov, Xctrl_cov, savePrefix):
    [Xctrlz, Xcasez, Xctrl_covz, Xcase_covz, Xcase_vcz, Xcase_ctrl_vcz, Xcase_vcsz, Xcase_ctrl_vcsz, Xcase_vctz,
     Xcase_ctrl_vctz] = corrections(Xcase, Xctrl, Xcase_cov, Xctrl_cov)
    [Tvc, Svc, TSvc, Tvc_tvc, Svc_svc, TSvc_tsvc, Tvct_s, Svcs_s, Tvct_Svcs_s, Tvct_tvc_s, Svcs_svc_s,
     Tvct_Svcs_tvc_svc_s] = makeVariableSets(Xcase_covz, Xcase_vcz, Xcase_vcsz, Xcase_vctz)
    [Tvc_ctrl, Svc_ctrl, TSvc_ctrl, Tvc_tvc_ctrl, Svc_svc_ctrl, TSvc_tsvc_ctrl, Tvct_s_ctrl, Svcs_s_ctrl,
     Tvct_Svcs_s_ctrl, Tvct_tvc_s_ctrl, Svcs_svc_s_ctrl, Tvct_Svcs_tvc_svc_s_ctrl] = makeVariableSets(Xcase_covz,
                                                                                                      Xcase_ctrl_vcz,
                                                                                                      Xcase_ctrl_vcsz,
                                                                                                      Xcase_ctrl_vctz)

    sets = ['Tvc', 'Svc', 'TSvc', 'Tvc_tvc', 'Svc_svc', 'TSvc_tsvc', 'Tvct_s', 'Svcs_s', 'Tvct_Svcs_s', 'Tvct_tvc_s',
            'Svcs_svc_s', 'Tvct_Svcs_tvc_svc_s']
    for n in range(len(sets)):
        with open(('/Users/lee_jollans/Documents/GitHub/ML_in_python/residfiles_all_210220/' + savePrefix + '_' + sets[
            n] + '.csv'), mode='w') as file:
            filewriter = csv.writer(file, delimiter=',')
            filewriter.writerows(eval(sets[n]))
        file.close()
        with open(('/Users/lee_jollans/Documents/GitHub/ML_in_python/residfiles_all_210220/' + savePrefix + '_' + sets[
            n] + '_ctrl.csv'), mode='w') as file:
            filewriter = csv.writer(file, delimiter=',')
            filewriter.writerows(eval(sets[n] + '_ctrl'))
        file.close()


def makeAndSave_noventricles(Xcase, Xctrl, Xcase_cov, Xctrl_cov, savePrefix, savedir):
    [Xctrlz, Xcasez, Xctrl_covz, Xcase_covz, Xcase_cz, Xcase_ctrl_cz, Xcase_csz, Xcase_ctrl_csz, Xcase_ctz,
     Xcase_ctrl_ctz] = corrections_noventricles(Xcase, Xctrl, Xcase_cov, Xctrl_cov)
    [Tc, Sc, TSc, Tc_tc, Sc_sc, TSc_tsc, Tct_s, Scs_s, Tct_Scs_s, Tct_tc_s, Scs_sc_s,
     Tct_Scs_tc_sc_s] = makeVariableSets_noventricles(Xcase_covz, Xcase_cz, Xcase_csz, Xcase_ctz)
    [Tc_ctrl, Sc_ctrl, TSc_ctrl, Tc_tc_ctrl, Sc_sc_ctrl, TSc_tsc_ctrl, Tct_s_ctrl, Scs_s_ctrl, Tct_Scs_s_ctrl,
     Tct_tc_s_ctrl, Scs_sc_s_ctrl, Tct_Scs_tc_sc_s_ctrl] = makeVariableSets_noventricles(Xcase_covz, Xcase_ctrl_cz,
                                                                                         Xcase_ctrl_csz, Xcase_ctrl_ctz)

    sets = ['Tc', 'Sc', 'TSc', 'Tc_tc', 'Sc_sc', 'TSc_tsc', 'Tct_s', 'Scs_s', 'Tct_Scs_s', 'Tct_tc_s', 'Scs_sc_s',
            'Tct_Scs_tc_sc_s']
    for n in range(len(sets)):
        with open((savedir + savePrefix + '_' + sets[n] + '.csv'), mode='w') as file:
            filewriter = csv.writer(file, delimiter=',')
            filewriter.writerows(eval(sets[n]))
        file.close()
        with open((savedir + savePrefix + '_' + sets[n] + '_ctrl.csv'), mode='w') as file:
            filewriter = csv.writer(file, delimiter=',')
            filewriter.writerows(eval(sets[n] + '_ctrl'))
        file.close()


