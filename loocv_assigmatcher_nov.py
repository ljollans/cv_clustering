# load LOOCV cluster assignments and get a consensus cluster naming
import sys
import os
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy import sparse as sp

from utils import select_trainset, percent_overlap_vectors, get_pac, rand_score_withnans, max_min_val_check, \
    contingency_matrix, ecdf


def return_train_data(X, mainfold, subfold):
    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)
    trainset=select_trainset(cv_assignment, mainfold, subfold)
    return X[trainset,:]


def getloopcount(savestr, null, sets):
    # collect results from fold-wise run of GMM full covariance with LOOCV

    ex = np.zeros(shape=[2, len(sets), 16])
    for s in range(len(sets)):
        for fold in range(16):
            if null == 0:
                if os.path.isfile(savestr + str(sets[s]) + 'allcluslabels_fold' + str(fold) + '.pkl'):
                    ex[0, s, fold] = 1
                if os.path.isfile(savestr + str(sets[s]) + 'allcluslabels_ctrl_fold' + str(fold) + '.pkl'):
                    ex[1, s, fold] = 1
            else:
                if os.path.isfile(savestr + str(sets[s]) + 'allcluslabels_null_fold' + str(fold) + '.pkl'):
                    ex[0, s, fold] = 1
                if os.path.isfile(savestr + str(sets[s]) + 'allcluslabels_null_ctrl_fold' + str(fold) + '.pkl'):
                    ex[1, s, fold] = 1
    ex1 = np.sum(ex, axis=2)
    print(ex1)
    return ex1


def get_k_from_bic(savedir, ex1, null,sets):

    sil = np.full([398, 8, 4, 4, 2, len(sets)], np.nan)
    cal = np.full([398, 8, 4, 4, 2, len(sets)], np.nan)
    bic = np.full([398, 8, 4, 4, 2, len(sets)], np.nan)
    best_k_loocv = np.full([4, 4, 398, 2, len(sets)], np.nan)
    best_k_sf = np.full([4, 4, 2, len(sets)], np.nan)
    best_k_mf = np.full([4, 2, len(sets)], np.nan)
    best_sil = np.full([4, 2, len(sets)], np.nan)
    best_cal = np.full([4, 2, len(sets)], np.nan)
    best_bic = np.full([4, 2, len(sets)], np.nan)
    pct_agreement_k = np.zeros(shape=[4, 4, 2, len(sets)])

    for s in range(len(sets)):
        for ctr in range(2):
            if null == 1:
                if ctr == 1:
                    str2add = '_null_ctrl'
                else:
                    str2add = '_null'
            else:
                if ctr == 1:
                    str2add = '_ctrl'
                else:
                    str2add = ''
            if ex1[ctr, s] == 16:
                fold = -1
                for mainfold in range(4):
                    for subfold in range(4):
                        fold += 1

                        pkl_filename1 = (savedir + sets[s] + 'SIL' + str2add + '_fold' + str(fold) + '.pkl')
                        pkl_filename2 = (savedir + sets[s] + 'CAL' + str2add + '_fold' + str(fold) + '.pkl')
                        pkl_filename3 = (savedir + sets[s] + 'BIC' + str2add + '_fold' + str(fold) + '.pkl')

                        with open(pkl_filename1, 'rb') as file:
                            sil[:, :, mainfold, subfold, ctr, s] = pickle.load(file)
                        with open(pkl_filename2, 'rb') as file:
                            cal[:, :, mainfold, subfold, ctr, s] = pickle.load(file)
                        with open(pkl_filename3, 'rb') as file:
                            bic[:, :, mainfold, subfold, ctr, s] = pickle.load(file)

                        for loocv in range(bic.shape[0]):
                            if len(np.where(np.isnan(bic[loocv, :, mainfold, subfold, ctr, s]))[0]) == 0:
                                best_k_loocv[mainfold, subfold, loocv, ctr, s] = np.where(
                                    bic[loocv, :, mainfold, subfold, ctr, s] == np.nanmin(
                                        bic[loocv, :, mainfold, subfold, ctr, s]))[0]

                        bic_mean_all_loocv = np.nanmean(bic[:, :, mainfold, subfold, ctr, s], axis=0)

                        best_k_sf[mainfold, subfold, ctr, s] = \
                            np.where(bic_mean_all_loocv == np.nanmin(bic_mean_all_loocv))[0]

                    bic_mean_all_loocv_sf = np.nanmean(np.nanmean(bic[:, :, mainfold, :, ctr, s], axis=0), axis=1)
                    best_k_mf[mainfold, ctr, s] = \
                    np.where(bic_mean_all_loocv_sf == np.nanmin(bic_mean_all_loocv_sf))[0]

                    best_bic[mainfold, ctr, s] = np.nanmean(
                        bic[:, best_k_mf[mainfold, ctr, s].astype(int), mainfold, :, ctr, s])
                    best_sil[mainfold, ctr, s] = np.nanmean(
                        sil[:, best_k_mf[mainfold, ctr, s].astype(int), mainfold, :, ctr, s])
                    best_cal[mainfold, ctr, s] = np.nanmean(
                        cal[:, best_k_mf[mainfold, ctr, s].astype(int), mainfold, :, ctr, s])

                for mainfold in range(4):
                    for subfold in range(4):
                        ct_match = len(
                            np.where(best_k_loocv[mainfold, subfold, :, ctr, s] == best_k_mf[mainfold, ctr, s])[0])
                        ct_nonnan = len(np.where(np.isfinite(best_k_loocv[mainfold, subfold, :, ctr, s]))[0])
                        pct_agreement_k[mainfold, subfold, ctr, s] = (ct_match * 100) / ct_nonnan

    return best_sil, best_cal, best_bic, best_k_mf, best_k_sf, best_k_loocv, sil, cal, bic, pct_agreement_k


def get_clusassignments_from_LOOCV(set, mainfold, subfold,sets,savedir,null,ctr):
    fold_number = (4 * mainfold) + subfold

    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)

    if null==1:
        pkl_filename = (savedir + sets[set] + 'allcluslabels_null_fold' + str(fold_number) + '.pkl')
    elif ctr==1:
        pkl_filename = (savedir + sets[set] + 'allcluslabels_ctrl_fold' + str(fold_number) + '.pkl')
    else:
        pkl_filename = (savedir + sets[set] + 'allcluslabels_fold' + str(fold_number) + '.pkl')
    with open(pkl_filename, "rb") as file:
        Aorig = pickle.load(file)
    trainset = select_trainset(cv_assignment, mainfold, subfold)
    A1 = Aorig[trainset, :, :]
    A2 = A1[:, :, trainset]
    return A2


def plot_bic_violin(BIC, mainfold, subfold, ctr, set):
    bic = pd.DataFrame({}, columns=['bic', 'k', 'ppt'])
    for nclus in range(BIC.shape[1]):
        tmp_df = pd.DataFrame(
            {'bic': np.squeeze(BIC[:, nclus, mainfold, subfold, ctr, set]),
             'k': np.ones(shape=[398]) * nclus + 2,
             'ppt': np.arange(398)},
            columns=['bic', 'k', 'ppt'])
        bic = bic.append(tmp_df)
    sns.violinplot('k', 'bic', data=bic)


def n_clus_retrieval_chk(cluster_assignments, cocluster_ratio, uniqueness_pct):
    n_maxcluster = np.zeros(shape=[cluster_assignments.shape[1]])
    pacs = np.zeros(shape=[cluster_assignments.shape[1]])
    for ngroups in range(cluster_assignments.shape[1]):
        cluster_assignments_k = cluster_assignments[:, ngroups, :]
        co_cluster_count = get_co_cluster_count(cluster_assignments_k)
        try:
            pacs[ngroups] = get_pac(co_cluster_count)
        except:
            pacs[ngroups] = 0.0

        maxmatches, findmax = get_maxmatches(cluster_assignments_k, co_cluster_count, cocluster_ratio)

        final_assignment = get_final_assignment(maxmatches, uniqueness_pct)

        n_maxcluster[ngroups] = (len(final_assignment))
    u_cr = np.unique(n_maxcluster)
    cr_count = [len(np.where(n_maxcluster == i)[0]) for i in u_cr]
    max_cr = np.max(cr_count)
    find_max_cr = np.where(cr_count == max_cr)

    return n_maxcluster, pacs, u_cr[find_max_cr[0][0]]

def n_clus_retrieval_grid(cluster_assignments):
    n_params=5
    cocluster_ratio = np.linspace(2/3,1,n_params)
    uniqueness_pct = np.linspace(0,100/3,n_params)

    n_maxcluster = np.zeros(shape=[n_params,n_params,cluster_assignments.shape[1]])
    pacs = np.zeros(shape=[cluster_assignments.shape[1]])
    for ngroups in range(cluster_assignments.shape[1]):
        cluster_assignments_k = cluster_assignments[:, ngroups, :]
        co_cluster_count = get_co_cluster_count(cluster_assignments_k)
        try:
            pacs[ngroups] = get_pac(co_cluster_count)
        except:
            pacs[ngroups] = 0.0
        for cocluster_loop in range(n_params):
            maxmatches, findmax = get_maxmatches(cluster_assignments_k, co_cluster_count, cocluster_ratio[cocluster_loop])
            for uniqueness_loop in range(n_params):
                final_assignment = get_final_assignment(maxmatches, uniqueness_pct[uniqueness_loop])

                n_maxcluster[cocluster_loop,uniqueness_loop,ngroups] = (len(final_assignment))

    return n_maxcluster, pacs, cocluster_ratio, uniqueness_pct


def k_workup_mainfold(mainfold, set, sets, savedir):

    cv_assignment_dir = "/Users/lee_jollans/Documents/GitHub/ML_in_python/export_251019/"
    with open((cv_assignment_dir + "CVassig398.csv"), "r") as f:
        reader = csv.reader(f, delimiter=",")
        cv_assignment = np.array(list(reader)).astype(float)

    ex1 = getloopcount(savedir, 0)
    [bestSIL, bestCAL, bestnclus_mf, bestnclus_sf, SIL, CAL, BIC] = get_k_from_bic(savedir, ex1, 0)
    fig = plt.figure();
    plt.rc('font', size=8)
    ctr = 0
    for subfold in range(4):
        ctr += 1;
        plt.subplot(4, 3, ctr)
        plot_bic_violin(BIC, mainfold, subfold);
        plt.title('BIC')

        # load data
        pkl_filename = (savedir + sets[set] + 'allcluslabels_fold' + str(
            (mainfold * 4) + subfold) + '.pkl')
        with open(pkl_filename, "rb") as file:
            Aorig = pickle.load(file)
        trainset = select_trainset(cv_assignment, mainfold, subfold)
        A1 = Aorig[trainset, :, :]
        A2 = A1[:, :, trainset]

        # cluster "saturation" and PAC score
        cr, pac, plateau_cr = n_clus_retrieval_chk(A2, .8, 25)

        ctr += 1;
        plt.subplot(4, 3, ctr);
        plt.title('number of unique clusters across LOOCV')
        plt.plot(cr);
        plt.xlabel('k');
        plt.xticks(np.arange(len(cr)), np.arange(len(cr)) + 2)
        ctr += 1;
        plt.subplot(4, 3, ctr);
        plt.title('proportion of ambiguous clustering')
        plt.plot(pac);
        plt.xlabel('k');
        plt.xticks(np.arange(len(cr)), np.arange(len(cr)) + 2)
    plt.subplots_adjust(0.125, 0.1, 0.9, 0.9, 0.1, 0.45)
    plt.show()


def get_co_cluster_count(cluster_assignments):
    # returns ratio of co_clustering for every pair of observations
    # input: cluster_assignment (i x N matrix) for i iterations and N observations
    # output: co_cluster_count (NxN matrix) with values between
    # 0 (never cluster together) and 1 (always cluster together)

    # e.g. co_cluster_count[n1,n2] is the ratio of times cluster_assignments[i,n1]==cluster_assignment[i,n2]) within
    # every instance of n1 and n2 being assigned a group in the same i

    co_cluster_count = np.full([cluster_assignments.shape[1], cluster_assignments.shape[1]], 0.00)
    for n1 in range(cluster_assignments.shape[1]):
        for n2 in range(cluster_assignments.shape[1]):
            if n1 < n2:
                co_cluster_sum = len(np.where(cluster_assignments[:, n1] == cluster_assignments[:, n2])[0])
                co_assignment_instances = cluster_assignments.shape[0] - len(
                    np.unique(
                        np.append(
                            np.where(np.isnan(cluster_assignments[:, n1]))[0],
                            np.where(np.isnan(cluster_assignments[:, n2]))[0],
                        )
                    )
                )
                co_cluster_count[n1, n2] = co_cluster_sum / co_assignment_instances

    return co_cluster_count


def get_maxmatches(cluster_assignments, co_cluster_count, ratio):
    # returns assignments for pairs of observations in cluster_assignments where ratio of
    # co_clustering is >= ratio
    # input: cluster_assignment (i x N matrix) for i iterations and N observations,
    #        co_cluster_count (NxN matrix) with values between 0 and 1 (always cluster together)
    #        ratio: value between 0 and 1 for the minimum ratio of co-clustering for assignments to be returned
    # output: findmax: indices of f observation pairs that co-cluster >= ratio
    #         maxmatches (i x f) matrix containing the overlapping assignments for the pairs in findmax

    max_match = ratio  # np.max(co_cluster_count) * ratio_of_max_as_lower
    findmax = np.where(co_cluster_count >= max_match)
    #print(
    #    "highest match frequency=",
    #    str(max_match),
    #    "occurring for",
    #    str(len(findmax[0])),
    #    "pairs.",
    #)
    maxmatches = np.full([cluster_assignments.shape[0], len(findmax[0]), ], np.nan)
    for n in range(len(findmax[0])):
        a1 = findmax[0][n]
        a2 = findmax[1][n]
        matches = np.where(cluster_assignments[:, a1] == cluster_assignments[:, a2])[0]
        maxmatches[matches, n] = cluster_assignments[matches, a1]
    return maxmatches, findmax


def get_final_assignment(maxmatches, max_pct):
    # for the assignments in maxmatches identifies the number of unique assignment patterns
    # if the overlap of an assignment pattern with any already identified pattern is less than ratio
    # then the pattern is treated as a new unique assignment pattern
    # input: maxmatches:  (i x f) matrix containing the f assignments from i iterations to be evaluated
    #        max_pct: the maximum perecentage overlap an assignment pattern can have with another to still be
    #                 considered unique
    # output: final_assignment: (i x u) matrix with the u unique assignment patterns

    final_assignment = []
    for n in range(maxmatches.shape[1]):
        if len(final_assignment) > 0:
            tmp_match_pct = np.zeros(len(final_assignment))
            for clus in range(len(final_assignment)):
                tmp_match_pct[clus] = percent_overlap_vectors(
                    maxmatches[:, n], final_assignment[clus]
                )
            if np.max(tmp_match_pct) < max_pct:
                final_assignment.append(maxmatches[:, n])
        else:
            final_assignment.append(maxmatches[:, n])
    return final_assignment


def match_assignments_to_final_assignments(cluster_assignments, final_assignment):
    match_pct = np.zeros(shape=[cluster_assignments.shape[1], len(final_assignment)])
    for n in range(cluster_assignments.shape[1]):
        for clus in range(len(final_assignment)):
            match_pct[n, clus] = percent_overlap_vectors(
                cluster_assignments[:, n], final_assignment[clus]
            )
    return match_pct


def get_aggregated_patterns(final_assignment, meet_fit_requirements, match_pct, A):
    n_groups = (len(final_assignment))
    aggregated_patterns_lists = []
    aggregated_patterns_arrays = []
    for p in range(n_groups):
        aggregated_patterns_lists.append([])
        aggregated_patterns_arrays.append([])

    for ppt in meet_fit_requirements:
        ppt_fit = np.where(match_pct[ppt, :] == np.max(match_pct[ppt, :]))[0]
        aggregated_patterns_lists[ppt_fit[0]].append(A[:, ppt])

    patterns_per_group = [len(aggregated_patterns_lists[i]) for i in range(n_groups)]
    n_iterations = A.shape[0]
    # restructure
    for g in range(n_groups):
        aggregated_patterns_arrays[g] = np.full([n_iterations, patterns_per_group[g]], np.nan)
        for p in range(patterns_per_group[g]):
            aggregated_patterns_arrays[g][:, p] = aggregated_patterns_lists[g][p]

    return aggregated_patterns_arrays


def recode_iteration_assignments(aggregated_patterns_arrays, k):
    n_groups = len(aggregated_patterns_arrays)
    patterns_per_group = [len(aggregated_patterns_arrays[i]) for i in range(n_groups)]
    n_iterations = aggregated_patterns_arrays[0].shape[0]
    corresponding_cluster = np.full([n_iterations, n_groups], np.nan)
    for i in range(n_iterations):
        tmp_match = np.full([n_groups, k], 0.00)
        for g in range(n_groups):
            crit = aggregated_patterns_arrays[g][i, :]
            u_crit = np.unique(crit)
            for n in range(len(u_crit)):
                c = (len(np.where(crit == u_crit[n])[0]) / len(crit))
                if np.isfinite(u_crit[n]):
                    tmp_match[g, u_crit[n].astype(int)] = c
        corresponding_cluster[i, :] = maxmatch_from_contingency_matrix(tmp_match, 0.5)
    return corresponding_cluster


def maxmatch_from_contingency_matrix(con_mat, minmatch):
    corresponding_cluster = np.full([con_mat.shape[0]], np.nan)
    while np.max(con_mat) >= minmatch:
        max_match = np.where(con_mat == np.max(con_mat))
        corresponding_cluster[max_match[0][0]] = max_match[1][0]
        con_mat[max_match[0][0], :] = np.zeros(shape=[con_mat.shape[1]])
        con_mat[:, max_match[1][0]] = con_mat[:, max_match[1][0]] - con_mat[:, max_match[1][0]]
    return corresponding_cluster


def rand_score_comparison(A, consensus_label):
    rand_all = np.full([A.shape[0], A.shape[0]], np.nan)
    for a1 in range(A.shape[0]):
        for a2 in range(A.shape[0]):
            if a2 > a1:
                rand_all[a1, a2] = rand_score_withnans(A[a1, :], A[a2, :])

    # plt.imshow(rand_all); plt.colorbar(); plt.show()
    print('median rand score between iterations:', np.nanmedian(rand_all))
    print('mean rand score between iterations:', np.nanmean(rand_all))
    print('smallest rand score between iterations:', np.nanmin(rand_all))

    rand_aftermatch = [rand_score_withnans(consensus_label, A[i, :]) for i in range(A.shape[0])]
    print('median rand score between the consensus assignment and the iteration assignments:', np.nanmedian(rand_aftermatch))
    print('mean rand score between the consensus assignment and the iteration assignments:', np.nanmean(rand_aftermatch))
    print('smallest rand score between the consensus assignment and the iteration assignments:', np.nanmin(rand_aftermatch))
    return rand_all, rand_aftermatch


def collect_betas_for_corresponding_clus(corresponding_cluster, betas):
    n_iterations, n_groups = corresponding_cluster.shape
    aggregated_betas = np.full([betas.shape[2], n_groups], np.nan)

    new_betas_list = []
    new_betas_array = []
    for groups in range(n_groups):
        new_betas_list.append([])
        new_betas_array.append([])
        for i in range(n_iterations):
            crit = corresponding_cluster[i, groups]
            if np.isfinite(crit):
                new_betas_list[groups].append(betas[i, crit.astype(int), :])
        new_betas_array[groups] = np.full([betas.shape[2], len(new_betas_list[groups])], np.nan)
        for i in range(len(new_betas_list[groups])):
            new_betas_array[groups][:, i] = new_betas_list[groups][i]
        aggregated_betas[:, groups] = np.median(new_betas_array[groups], axis=1)

    return aggregated_betas, new_betas_array


def get_consensus_labels(A,k, verbose):

    # calculate co-clustering ratio for all observation pairs
    co_cluster_count = get_co_cluster_count(A)

    # choose the best matching pairs
    max_matches, max_match_locs = get_maxmatches(A, co_cluster_count, 0.8)

    # use the best pairs to identify unique assignment patterns corresponding to clusters
    final_assignment = get_final_assignment(max_matches, 25)
    if verbose==1:
        print('identified', len(final_assignment), 'unique patterns')

    # calculate how well all observations fit to each of these top assignment patterns
    match_pct = match_assignments_to_final_assignments(A, final_assignment)
    if verbose==1:
        #plt.hist(np.max(match_pct,axis=1)); plt.title('highest % overlap for assignments across the 6 iterations'); plt.show()
        print('median fit to best fitting pattern is',np.median(np.max(match_pct,axis=1)))

    # pick those observations that have a clear membership to one of the patterns
    meet_fit_requirements = max_min_val_check(match_pct, 50, 25)
    if verbose==1:
        print(len(meet_fit_requirements), 'observations passed the requirement for a good fit to a pattern')

    # sort those observations into the matching pattern and collect their assignments from all iterations
    aggregated_patterns_arrays = get_aggregated_patterns(final_assignment, meet_fit_requirements, match_pct, A)
    if verbose==1:
        num_p_per_g=[aggregated_patterns_arrays[g].shape[1] for g in range(len(aggregated_patterns_arrays))]
        print('number of observations collected for each pattern:',num_p_per_g)

    # for each iteration identify what cluster corresponds to the new patterns
    corresponding_cluster = recode_iteration_assignments(aggregated_patterns_arrays, k)

    # calculate consensus cluster assignment for all observations (not using betas)
    n_groups = corresponding_cluster.shape[1]
    n_iterations = A.shape[0]
    n_obs = A.shape[1]
    checkclus = np.zeros(shape=[n_groups, n_iterations, n_obs])
    consensus_label = np.full(n_obs, np.nan)
    for group in range(n_groups):
        for i in range(n_iterations):
            checkclus[group, i, np.where(A[i, :] == corresponding_cluster[i, group])[0]] = 1
    for ppt in range(n_obs):
        crit = np.sum(checkclus[:, :, ppt], axis=1)
        consensus_label[ppt] = np.where(crit == np.max(crit))[0][0]

    if verbose==1:
        randall,randaftermatch = rand_score_comparison(A, consensus_label)

    return consensus_label


def infer_iteration_clusmatch(consensus_labels, A):
    nclus=len(np.unique(consensus_labels))
    ni=A.shape[0]
    assignments = np.full([ni,nclus],np.nan)
    for i in range(ni):

        a=consensus_labels; au=np.unique(consensus_labels)

        b=A[i,:]; bu=np.unique(A[i,:])

        a=np.delete(a,np.where(np.isnan(b))[0]);
        b=np.delete(b,np.where(np.isnan(b))[0])
        a_u=np.unique(a);
        b_u=np.unique(b);

        tmp_conmat=contingency_matrix(a,b);

        conmat=np.full([len(au), len(bu)],np.nan)
        for m1 in range(len(a_u)):
            for m2 in range(len(b_u)):
                conmat[a_u[m1].astype(int),b_u[m2].astype(int)]=tmp_conmat[m1,m2]

        assignments[i,:]=maxmatch_from_contingency_matrix(conmat, 1)
    return assignments


def sort_into_clusters_argmax_ecdf(data, betas, mincdf=.8):
    Y = data.dot(betas)
    all_ys = np.full([Y.shape[0], Y.shape[1]], np.nan)
    for nclus in range(Y.shape[1]):
        xs, ys, idx = ecdf(Y[:, nclus])
        all_ys[idx, nclus] = ys
    argmax_assig = np.full(Y.shape[0], np.nan)
    for ppt in range(Y.shape[0]):
        crit = all_ys[ppt, :]
        if np.max(crit) > mincdf:
            argmax_assig[ppt] = np.where(crit == np.max(crit))[0][0]
    return argmax_assig, all_ys, Y


def pull_from_mod(mod):
    return mod.pac, mod.rand_all, mod.coph, mod.beta_aggregate, mod.beta_pre_aggregate, mod.cluster_ensembles_labels

def pull_from_mod2(mod):
    return mod.coph, mod.beta_aggregate, mod.beta_pre_aggregate, mod.cluster_ensembles_labels
