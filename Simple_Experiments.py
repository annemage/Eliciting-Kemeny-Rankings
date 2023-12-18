import numpy as np
import pandas as pd
import itertools
import math
import copy
import random
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from Preferences import generate_profile, build_matrix_from_profile, generate_profile_Mallows, \
    generate_sp_vote_profile_uar

###################################################
### Computing Kemeny Rankings (with Integer Linear Program formulation)
###################################################

def exact_kemeny(Q):
    k = len(Q)

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SCIP')

    # Define 0,1 variables for every ordered pair of arms
    x = {}
    for i in range(k):
        for j in range(k):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

    # Define Constraints ensuring only one tuple (a,b) or (b,a) is selected: x_i_j + x_j_i = 1
    for i, j in itertools.combinations(range(k), 2):
        constraint = solver.RowConstraint(1, 1)
        constraint.SetCoefficient(x[i, j], 1)
        constraint.SetCoefficient(x[j, i], 1)

    # Define constraints ensuring transitivity: 1 <= x_i_j + x_j_k + x_k_i <= 2
    for i, j, h in itertools.combinations(range(k), 3):
        constraint = solver.RowConstraint(1, 2)
        constraint.SetCoefficient(x[i, j], 1)
        constraint.SetCoefficient(x[j, h], 1)
        constraint.SetCoefficient(x[h, i], 1)

    # Define objective
    objective = solver.Objective()
    for i in range(k):
        for j in range(k):
            if i != j:
                objective.SetCoefficient(x[i, j], Q[j][i])
    objective.SetMinimization()

    status = solver.Solve()

    # Set Kemeny Score and Kemeny Ranking from ILP solution
    kemeny_score = solver.Objective().Value()
    kemeny_ranking = -1 * np.ones(k)
    for i in range(k):
        sum = 0
        for j in range(k):  # for every arm compute the number of lower ranked arms (in "sum")
            if i != j:
                sum += x[i, j].solution_value()
        kemeny_ranking[int(k - sum - 1)] = i  # the arm i is placed according to the number of lower ranked arms

    return kemeny_score, kemeny_ranking

###################################################
### x, y, and confidences as indicated in the paper
###################################################

def x_value(rho, k):
    return k * (k - 1) / rho


def y_value(delta, k):
    return math.log(delta / (k * (k - 1)))


def confidence_wo_replacement(n, pulls, y):
    if pulls == 0:
        return 2
    if pulls > n / 2:
        c = math.sqrt(-(n - pulls) * (pulls + 1) * y / (2 * pulls * pulls * n))
    else:
        c = math.sqrt(-(n - pulls + 1) * y / (2 * pulls * n))
    return round(c, 5)


def confidence_w_replacement(n, pulls, y):
    if pulls == 0:
        return 2
    c = math.sqrt(-y / (2 * pulls))
    return round(c, 5)


###################################################
### Pruning Confidence Intervals: essentially Algorithm 3 from the paper
###################################################

def pruning(Q_hat, C):
    k = len(Q_hat)
    # do the pruning based on bounds [0,1] for q_ij:
    # hat(q_ij) + c_ij <= 1    <==>      hat(q_ji) - c_ij >= 0
    #                          <==>      hat(q_ji)        >= c_ij
    C_dot = np.minimum(C, Q_hat.transpose())

    # do the pruning based on bounds [0,1] for q_ij:
    # hat(q_ij) + c_ij >= 0    <==>      c_ij >= - hat(q_ij)
    C_dot = np.maximum(C_dot, - Q_hat)

    # do the pruning based on triangle inequality
    pruning_happened = True
    while pruning_happened:
        # while not np.equal(C_temp, C_dot).all():  # check if last round of pruning changed any values
        pruning_happened = False
        C_temp = copy.deepcopy(C_dot)
        for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms

            # triangle-inequality style pruning
            C_dot[i][j] = min(C_temp[i][j], min([Q_hat[i][l] + C_temp[i][l] + Q_hat[l][j] + C_temp[l][j] - Q_hat[i][j]
                                                 for l in range(k) if l != j and l != i]))
            C_dot[i][j] = round(C_dot[i][j], 5)   # round up to five digits to avoid numerical difficulties

            # readjust values to respect bounds 0 <= C_dot[i][j] + Q_hat[i][j] <= 1
            C_dot[i][j] = max(- Q_hat[i][j], C_dot[i][j])
            C_dot[i][j] = min(Q_hat[j][i], C_dot[i][j])

            # triangle-inequality style pruning
            C_dot[j][i] = min(C_temp[j][i], min([Q_hat[j][l] + C_temp[j][l] + Q_hat[l][i] + C_temp[l][i] - Q_hat[j][i]
                                                 for l in range(k) if l != j and l != i]))
            C_dot[j][i] = round(C_dot[j][i], 5)               # round up to five digits to avoid numerical difficulties

            # readjust values to respect bounds 0 <= C_dot[j][i] + Q_hat[j][i] <= 1
            C_dot[j][i] = max(- Q_hat[j][i], C_dot[j][i])
            C_dot[j][i] = min(Q_hat[i][j], C_dot[j][i])

        assert sum([C_temp[i][j] for (i, j) in itertools.product(range(k), range(k)) if i != j]) \
               >= sum([C_dot[i][j] for (i, j) in itertools.product(range(k), range(k)) if i != j]), \
            "my pruning makes bounds larger?!?"
        if not np.equal(C_temp, C_dot).all():
            pruning_happened = True
            # print('done some pruning')
    return C_dot



###################################################
### SAMPLING STRATEGIES
###################################################

def uniform_sampling(Q, n, Q_hat, C, N, conf_update, y):
    k = len(Q)
    min_i = 0
    min_j = 1
    amn = N[0][1]
    for (i, j) in itertools.combinations(range(k), 2):
        if N[i][j] < amn:
            min_i = i
            min_j = j
            amn = N[min_i][min_j]
    if conf_update.__name__ == 'confidence_wo_replacement':
        assert amn < n, 'it seems I have already asked the voters everything and still want to sample more!'
    return min_i, min_j


def opportunistic_sampling(Q, n, Q_hat, C, N, conf_update, y):
    k = len(Q)
    max_i = 0
    max_j = 1
    amx = C[max_i][max_j] + C[max_j][max_i]
    for (i, j) in itertools.combinations(range(k), 2):
        if C[i][j] + C[j][i] > amx:
            # when sampling without replacement we can only sample from arms that we haven't asked all n voters about already
            if conf_update.__name__ != 'confidence_wo_replacement' or N[i][j] < n:
                max_i = i
                max_j = j
                amx = C[i][j] + C[j][i]
    if conf_update.__name__ == 'confidence_wo_replacement':
        assert N[max_i][
                   max_j] < n, f'it seems I have already asked all voters about arms {max_i} {max_j} and still want to sample more!'
    return max_i, max_j


def optimistic_sampling(Q, n, Q_hat, C, N, conf_update, y):
    k = len(Q)
    # if there exists a pair that has not been pulled often enough to get meaningfull confidence intervals - pull it!
    for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms
        if conf_update(n, N[i][j], y) >= 1:
            return (i, j)

    # otherwise ...
    best_i = None
    best_j = None
    max_val = 0
    for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms
        # when sampling without replacement we can only sample from arms that we haven't asked all n voters about already
        if conf_update.__name__ != 'confidence_wo_replacement' or N[i][j] < n:
            # if conf_update.__name__ == 'confidence_wo_replacement': print('good to go, got only ', N[i][j], 'pulls yet')
            ci_diffs = np.zeros(2)
            N_temp = copy.deepcopy(N)
            N_temp[i][j] += 1
            N_temp[j][i] += 1
            for smpl in range(2): # Simulate sampling and pruning
                Q_tmp = copy.deepcopy(Q_hat)
                Q_tmp[i][j] = (Q_tmp[i][j] * (N_temp[i][j] - 1) + smpl) / N_temp[i][j]
                Q_tmp[j][i] = 1 - Q_tmp[i][j]
                C_tmp = [[conf_update(n, N_temp[i][j], y) for j in range(k)] for i in range(k)]
                # build the confidence bounds by the chosen type of confidence bounds
                C_tmp = pruning(Q_tmp, C_tmp)
                ci_diffs[smpl] = sum(
                    [abs(conf_update(n, N[i][j], y) - C_tmp[i][j]) for (i, j) in itertools.product(range(k), range(k)) if i != j])

            # keep track of which arms are best to pull
            if max_val < max(ci_diffs):
                best_i = i
                best_j = j
                max_val = max(ci_diffs)
            assert conf_update(n, N[i][j], y) >= 1 or min(ci_diffs) > 0, f'it seems some sample outcome for the ' \
                                                                              f'two arms {i}, {j} will lead to ' \
                                                                              f'{min(ci_diffs)} change in the ' \
                                                                              f'confidence intervals! These arms have '\
                                                                              f'been pulled {N[i][j]} times and the ' \
                                                                              f'new confidence should be at most ' \
                                                                              f'{conf_update(n, N_temp[i][j], y)} ' \
                                                                              f'whereas the old one was ' \
                                                                              f'{C[i][j]} '

    # if none of the arm pairs are making any difference in the confidence intervals, just sample an arm with min
    # number pulls
    # note that this case can only occur when
    # 1) there are arms that haven't been pulled often (more than twice for small k), because then the confidence
    # intervals are too large and will be croped to 1 --> thus making no difference to the 0 pull case
    # 2) the intervals were already cropped so much, that the new confidence bounds just reach the same cropping
    if max_val == 0:
        return uniform_sampling(Q, n, Q_hat, C, N, conf_update, y)

    return best_i, best_j


def pessimistic_sampling(Q, n, Q_hat, C, N, conf_update, y):
    k = len(Q)
    # if there exists a pair that has not been pulled often enough to get meaningfull confidence intervals - pull it!
    for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms
        if conf_update(n, N[i][j], y) >= 1:
            return (i, j)

    # otherwise ...
    best_i = None
    best_j = None
    max_val = 0
    for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms
        # when sampling without replacement we can only sample from arms that we haven't asked all n voters about already
        if conf_update.__name__ != 'confidence_wo_replacement' or N[i][j] < n:
            # print(i,j,'had been sampled',N[i][j],'many times before')
            # print('we are doing sampling with:', conf_update.__name__)
            ci_diffs = np.zeros(2)
            N_temp = copy.deepcopy(N)
            N_temp[i][j] += 1
            N_temp[j][i] += 1
            for smpl in range(2):  # Simulate sampling and pruning
                Q_tmp = copy.deepcopy(Q_hat)
                Q_tmp[i][j] = (Q_tmp[i][j] * (N_temp[i][j] - 1) + smpl) / N_temp[i][j]
                Q_tmp[j][i] = 1 - Q_tmp[i][j]
                C_tmp = [[conf_update(n, N_temp[i][j], y) for j in range(k)] for i in range(k)]
                # build the confidence bounds by the chosen type of confidence bounds
                # print('the confidences are at the moment', C_tmp)
                C_tmp = pruning(Q_tmp, C_tmp)
                ci_diffs[smpl] = sum(
                    [abs(conf_update(n, N[i][j], y) - C_tmp[i][j]) for (i, j) in itertools.product(range(k), range(k)) if i != j])
            # print('for the two sample outcomes, my confidence intervalls would decrease by', ci_diffs)

            # keep track of which arms are best to pull
            if max_val < min(ci_diffs):
                best_i = i
                best_j = j
                max_val = min(ci_diffs)
            assert conf_update(n, N[i][j], y) >= 1 or min(ci_diffs) > 0, f'it seems some sample outcome for the ' \
                                                                              f'two arms {i}, {j} will lead to ' \
                                                                              f'{min(ci_diffs)} change in the ' \
                                                                              f'confidence intervals! These arms have '\
                                                                              f'been pulled {N[i][j]} times and the ' \
                                                                              f'new confidence should be at most ' \
                                                                              f'{conf_update(n, N_temp[i][j], y)} ' \
                                                                              f'whereas the old one was ' \
                                                                              f'{C[i][j]} '

    # if none of the arm pairs are making any difference in the confidence intervals, just sample an arm with min
    # number pulls
    # note that this case can only occur when
    # 1) there are arms that haven't been pulled often (more than twice for small k), because then the confidence
    # intervals are too large and will be croped to 1 --> thus making no difference to the 0 pull case
    # 2) the intervals were already cropped so much, that the new confidence bounds just reach the same cropping
    if max_val == 0:
        return uniform_sampling(Q, n, Q_hat, C, N, conf_update, y)

    return best_i, best_j


def realistic_sampling(Q, n, Q_hat, C, N, conf_update, y):
    k = len(Q)
    # if there exists a pair that has not been pulled often enough to get meaningfull confidence intervals - pull it!
    for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms
        if conf_update(n, N[i][j], y) >= 1:
            return (i, j)

    # otherwise ...
    best_i = None
    best_j = None
    max_val = 0
    for (i, j) in itertools.combinations(range(k), 2):  # go through all pairs of arms
        # when sampling without replacement we can only sample from arms that we haven't asked all n voters about already
        if conf_update.__name__ != 'confidence_wo_replacement' or N[i][j] < n:
            ci_diffs = np.zeros(2)
            N_temp = copy.deepcopy(N)
            N_temp[i][j] += 1
            N_temp[j][i] += 1
            for smpl in range(2):   # Simulate sampling and pruning
                Q_tmp = copy.deepcopy(Q_hat)
                Q_tmp[i][j] = (Q_tmp[i][j] * (N_temp[i][j] - 1) + smpl) / N_temp[i][j]
                Q_tmp[j][i] = 1 - Q_tmp[i][j]
                C_tmp = [[conf_update(n, N_temp[i][j], y) for j in range(k)] for i in range(k)]
                # build the confidence bounds by the chosen type of confidence bounds
                C_tmp = pruning(Q_tmp, C_tmp)
                ci_diffs[smpl] = sum(
                    [abs(conf_update(n, N[i][j], y) - C_tmp[i][j]) for (i, j) in itertools.product(range(k), range(k)) if i != j])

            # keep track of which arms are best to pull
            if max_val < (Q_tmp[i][j] * ci_diffs[1] + (1 - Q_tmp[i][j]) * ci_diffs[0]):
                best_i = i
                best_j = j
                max_val = (Q_tmp[i][j] * ci_diffs[1] + (1 - Q_tmp[i][j]) * ci_diffs[0])
            assert conf_update(n, N[i][j], y) >= 1 or min(ci_diffs) > 0, f'it seems some sample outcome for the ' \
                                                                              f'two arms {i}, {j} will lead to ' \
                                                                              f'{min(ci_diffs)} change in the ' \
                                                                              f'confidence intervals! These arms have '\
                                                                              f'been pulled {N[i][j]} times and the ' \
                                                                              f'new confidence should be at most ' \
                                                                              f'{conf_update(n, N_temp[i][j], y)} ' \
                                                                              f'whereas the old one was ' \
                                                                              f'{C[i][j]} '

    # if none of the arm pairs are making any difference in the confidence intervals, just sample an arm with min
    # number pulls
    # note that this case can only occur when
    # 1) there are arms that haven't been pulled often (more than twice for small k), because then the confidence
    # intervals are too large and will be croped to 1 --> thus making no difference to the 0 pull case
    # 2) the intervals were already cropped so much, that the new confidence bounds just reach the same cropping

    if max_val == 0:
        return uniform_sampling(Q, n, Q_hat, C, N, conf_update, y)

    return best_i, best_j


def elicitation(Q, n, rho, delta, sample_strategy, apply_pruning, with_replacement):
    ks_dist = []                                        # the difference between optimal kemeny score and that of
                                                        # current approximate solution
    approx_bounds = []                                  # the current approximation bound

    ### TO REDUCE COMPUTATIONAL COST: Do not compute the Kemeny Score in every round
    # opt_ks = 0
    opt_ks, opt_kemeny_ranking = exact_kemeny(Q)        # the optimal solution for the instance
    k = len(Q)                                          # the number of arms
    N = np.zeros((k, k))                                # the numbers of pulls of arms
    Q_hat = 0.5 * np.ones((k, k))                       # our estimates of Q initialised to 0.5
    C = 0.5 * np.ones((k, k))                           # our confidence bounds initialised to 0.5
    approx_bound = sum([C[i][j] for (i, j) in itertools.product(range(k), range(k)) if i != j])
                                                        # some initial bound on the total length of confidence intervals
    prune_count = 0                                     # indicates weather pruning of confidence intervals was applied

    # compute the theoretical sample complexities for this case (for sampling with / without replacement)
    x = x_value(rho, n_arms)
    y = y_value(delta, n_arms)
    if with_replacement:
        sample_complexity_theory = math.ceil(-x * x * y / 2 * n_arms * (n_arms - 1) / 2)
    else:
        sample_complexity_theory = math.ceil(-(x * x * y * n - 2 * n) / (2 * n - x * x * y) * n_arms * (n_arms - 1) / 2)
        assert sample_complexity_theory <= k * (k - 1) * n, f'my theoretical sample complexity exceeds the complexity of asking all voters about all pairs of arms!'

    if not with_replacement:
        # establish for all pairs of arms i,j a list of "voter responses" that contains
        #   as many 1's as voters that prefer i to j and
        #   as many 0's as voters that prefer j to i
        shuffled_prefs = [[[1] * round(Q[i][j] * n) + [0] * (n - round(Q[i][j] * n)) for j in range(k)] for i in
                          range(k)]
        # shuffle to simulate the random order in which voters may be asked (sampling without replacement)
        for (i, j) in itertools.product(range(k), range(k)):
            if i != j:
                random.shuffle(shuffled_prefs[i][j])

    while approx_bound > rho:
        total_pulls = sum([N[i][j] for (i, j) in itertools.product(range(k), range(k)) if i != j]) / 2
        if total_pulls >= sample_complexity_theory:
            # assert sample_strategy.__name__ != "uniform_sampling", f'the {sample_strategy.__name__} takes {total_pulls} samples which is more than the {sample_complexity_theory} that are theoretically necessary! This is most peculiar, because confidences should be exactly as in our theoretical results!'
            break

        # if we sample without replacement but have already asked all voters about all pairs of arms, we need to stop
        if (not with_replacement) and (total_pulls >= k * (k - 1) * n / 2):
            break

        # find the next pair of arms to sample from and sample according to selected method
        if with_replacement:
            i, j = sample_strategy(Q, n, Q_hat, C, N, confidence_w_replacement, y)
            sample = np.random.binomial(1, Q[i][j])
        else:
            i, j = sample_strategy(Q, n, Q_hat, C, N, confidence_wo_replacement, y)
            # need to make convention that i < j for sampling without replacement such that samples do net get confused
            if i > j:
                h = i
                i = j
                j = h
            assert int(
                N[i][j]) < n, f'seems I am trying to take more samples than there are voters?! So far I sampled: {N}'
            assert len(shuffled_prefs[i][j]) > N[i][j], f'seems I have too little samples here: {shuffled_prefs[i][j]}'
            # print(k,i,j,N[i][j],n)
            sample = shuffled_prefs[i][j][int(N[i][j])]

        # update the sample averages and number of pulls
        Q_hat[i][j] = (Q_hat[i][j] * N[i][j] + sample) / (N[i][j] + 1)
        Q_hat[j][i] = 1 - Q_hat[i][j]
        N[i][j] += 1
        N[j][i] += 1

        # compute the confidence bounds according to sampling method based on current # pulls
        if with_replacement:
            C = [[confidence_w_replacement(n, N[i][j], y) for j in range(k)] for i in range(k)]
        else:
            C = [[confidence_wo_replacement(n, N[i][j], y) for j in range(k)] for i in range(k)]

        # prune the confidence bounds if applicable
        if apply_pruning:
            C_pruned = pruning(Q_hat, C)
            if not np.array_equal(C, C_pruned):
                prune_count += 1
                C = copy.deepcopy(C_pruned)

        # record the true Kemeny Score of an approximated Kemeny ranking at every sample step
        approx_bound = sum([C[i][j] for (i, j) in itertools.product(range(k), range(k)) if i != j])
        approx_bounds.append(approx_bound)
        ### TO REDUCE COMPUTATIONAL COST: Do not compute the Kemeny Score in every round
        approx_ks, approx_kemeny_ranking = exact_kemeny(np.add(Q_hat, C))
        # approx_ks = 0
        ks_dist.append(approx_ks - opt_ks)
    return ks_dist, approx_bounds, N, prune_count


def update_array_average(avg_lst, new_lst, n_pulls_old):  # assume len(avg_lst) >= len(new_lst)
    for i in range(len(new_lst)):
        avg_lst[i] = (avg_lst[i] * n_pulls_old + new_lst[i]) / (n_pulls_old + 1)
    return avg_lst


def bring_lists_to_same_length(lists):
    length = max([len(lists[i]) for i in range(len(lists))])  # the length of the longest list in lists
    for i in range(len(lists)):
        if len(lists[i]) < length:
            lists[i] = lists[i] + [0] * (length - len(lists[i]))
    return lists


########## EXPERIMENTS ##############

random.seed(24)
np.random.seed(24)

delta = 0.05  # ... with probability (1-delta)
n = 10  # number of voters (only influences the values in the Q-matrix and sample complexity when sampling without replacement)
n_instances = 10  # the number of instances we average over
max_arms = 9  # max number of arms
min_arms = 3  # min number of arms

#  Shall plots be saved?
save_plot = True
# Instances might be desired to be single-peaked or according to mallow's phi model (here phi is hard coded to be 0.2)
# ONLY ONE CAN BE TRUE!
single_peaked = False
mallows = False

# set up the file names for saving results
if mallows:
    filename_base1 = 'mallows_phi0_2-'
else:
    if single_peaked:
        filename_base1 = 'single_peaked(uar)-'
    else:
        filename_base1 = 'uniform_preferences-'
filename_base1 = filename_base1 + str(n_instances) + 'instance-n' + str(n) + '-delta' + str(round(delta, 2))
# \
#                + ' - rho' + str(round(rho, 2)) + ' - k' + str(n_arms) + '- wr-pruned' + str(no_pruning_inst_wr) + ' - wor-pruned' + str(no_pruning_inst_wor)
for n_arms in range(min_arms, max_arms + 1):
    rho = 0.1 * n_arms * (n_arms - 1) / 2  # estimated Kemeny score is <= opt_score + 0.1 * (worst case KS) ...
    # note that the worst case Kemeny score is k*(k-1)/2 (choosing the exact opposite ranking of what all voters want)
    # print("rho", rho)
    filename_base2 = filename_base1 + '-rho' + str(round(rho, 2)) + '-k' + str(n_arms)

    # compute the theoretical sample complexities for this case (for sampling with / without replacement)
    x = x_value(rho, n_arms)
    y = y_value(delta, n_arms)
    sample_complexity_theory_wr = math.ceil(-x * x * y / 2 * n_arms * (n_arms - 1) / 2)
    sample_complexity_theory_wor = math.ceil(-(x * x * y * n - 2 * n) / (2 * n - x * x * y) * n_arms * (n_arms - 1) / 2)
    print("x:", x)
    print("y:", y)
    print("theoretical sample complexity - with replacement:", sample_complexity_theory_wr)
    print("theoretical sample complexity - without replacement:", sample_complexity_theory_wor)

    for with_replacement in range(2):  # sampling with / without replacement
        if with_replacement:
            filename_base3 = filename_base2 + '-wr'
        else:
            filename_base3 = filename_base2 + '-wor'
        for with_pruning in range(2):  # sampling with / without pruning of confidence intervals
            if with_pruning:
                filename = filename_base3 + '-wp'
                strategies = [uniform_sampling, opportunistic_sampling, optimistic_sampling, pessimistic_sampling,
                              realistic_sampling]

            else:
                filename = filename_base3 + '-wop'
                filename = filename.replace('.', '_')
                strategies = [uniform_sampling]

            # set up dictionary to save theoretical sample complexity, true Kemeny score, and average outcomes for all sample strategies
            dict = {}
            if with_replacement:
                sample_complexity_theory = sample_complexity_theory_wr
            else:
                sample_complexity_theory = sample_complexity_theory_wor
            dict["theoretical sample complexity"] = sample_complexity_theory
            dict["avg_true_KS"] = 0
            for sample_strategy in strategies:
                dict[sample_strategy.__name__] = {
                    # the average deviation to the optimal Kemeny scores for all instances
                    "avg_KS": 0,
                    # the average number of samples needed to reach approx bound rho under pruning
                    "avg_sample_complexity": 0,
                    # the sample complexities of all instances
                    "sample_complexities": np.zeros(n_instances),
                    # the number of instances in which pruning was applied to some effect
                    "no_pruning_inst": 0,
                    # the difference between optimal kemeny score and that of current approximate solution (over time steps)
                    "avg_ks_dist": np.zeros(sample_complexity_theory),
                    # the current approximation bound (over time steps)
                    "avg_approx_bounds": np.zeros(sample_complexity_theory)}

            # going through n_instances randomly generated instances
            for inst_idx in range(n_instances):
                print('working on instance nr', inst_idx)
                #########################
                # generate an instance - dependent on the type of preferences
                #########################
                if mallows:  # Mallows models are generated here with phi = 0.2
                    P = generate_profile_Mallows(n, n_arms, 0.2)
                else:
                    if single_peaked:
                        P = generate_sp_vote_profile_uar(n, n_arms)
                    else:
                        P = generate_profile(n, n_arms)
                Q = build_matrix_from_profile(P)
                # print('instance', Q)

                # get the optimal solution for the instance
                opt_ks, opt_kemeny_ranking = exact_kemeny(Q)
                dict["avg_true_KS"] += opt_ks  # update average of true Kemeny scores

                for sample_strategy in strategies:
                    print('Sampling', sample_strategy.__name__, 'with_pruning=', with_pruning, 'with_replacement=',
                          with_replacement)
                    # Set up collections of averages
                    #########################
                    # DO THE SAMPLING
                    #########################
                    # elicitation output:
                    # ks_dist = the differences in Kemeny score to the true Kemeny score over time,
                    # approx_bounds = the total length of confidence bounds over time,
                    # N = the final number of pulls of each pair of arms (matrix),
                    # prune_count = the number of iterations in which pruning decreased confidence intervals
                    ks_dist, approx_bounds, N, done_pruning \
                        = elicitation(Q, n, rho, delta, sample_strategy, with_pruning, with_replacement)
                    print('done pulls\n', N)

                    # keep track of the total number of samples taken
                    sample_complexity = int(sum([N[i][j] for (i, j) in itertools.combinations(range(n_arms), 2)]))
                    print('--> elicitation done after', sample_complexity, 'samples')
                    dict[sample_strategy.__name__]["avg_sample_complexity"] += sample_complexity
                    dict[sample_strategy.__name__]["sample_complexities"][inst_idx] = sample_complexity

                    # keep track of the number of instances in which pruning happened
                    if done_pruning > 0: dict[sample_strategy.__name__]["no_pruning_inst"] += 1

                    # an assert to check that the sample complexity is not larger than the theoretical one...
                    # assert sample_complexity <= sample_complexity_theory, "it appears that we need more samples than " \
                    #                                                       "theoretically necessary! The number of " \
                    #                                                       "samples taken is " + str(sample_complexity)

                    # keep track of average Kemeny score differences, and approximation bounds over time
                    dict[sample_strategy.__name__]["avg_ks_dist"][0:sample_complexity] += ks_dist
                    dict[sample_strategy.__name__]["avg_approx_bounds"][0:sample_complexity] += approx_bounds

                    # keep track of average Kemeny score differences (at end)
                    dict[sample_strategy.__name__]["avg_KS"] += ks_dist[-1]

            # make averages
            dict["avg_true_KS"] = dict["avg_true_KS"] / n_instances
            for sample_strategy in strategies:
                dict[sample_strategy.__name__]["avg_KS"] = dict[sample_strategy.__name__]["avg_KS"] / n_instances
                dict[sample_strategy.__name__]["avg_sample_complexity"] = dict[sample_strategy.__name__][
                                                                              "avg_sample_complexity"] / n_instances
                for min_no_samples in range(sample_complexity_theory):
                    # count the number of instances for which we actually had >= min_no_samples number of samples,
                    # i.e. added something to avg_ks_dist and avg_approx_bounds at entry [min_no_samples]
                    divisor = sum(1 for x in range(n_instances) if
                                  dict[sample_strategy.__name__]["sample_complexities"][x] >= min_no_samples)
                    if divisor > 0:
                        dict[sample_strategy.__name__]["avg_ks_dist"][min_no_samples] /= divisor
                        dict[sample_strategy.__name__]["avg_approx_bounds"][min_no_samples] /= divisor

            # set up pandas DataFrame and save to csv
            df = pd.DataFrame()
            df = df.from_dict(data=dict, orient='index')
            df.reset_index(inplace=True)
            df.to_csv(filename + '.csv')
            # print(df)
            # print(strategies)

            # plot confidence_bounds and save plot
            cb_png_filename = 'confidence_bounds-' + filename + '.png'
            cb_lists = [dict[sample_strategy.__name__]["avg_approx_bounds"] for sample_strategy in strategies]
            cb_lists = bring_lists_to_same_length(cb_lists)
            x = range(len(cb_lists[0]))
            plt.xlabel('# samples')
            plt.ylabel('confidence bounds')
            plt.title('averages for ' + filename)
            for i in range(len(cb_lists)):
                plt.plot(x, cb_lists[i], label=strategies[i].__name__)
            plt.plot(x, [rho]*len(cb_lists[0]), label='rho')
            plt.legend()
            plt.savefig(cb_png_filename)
            plt.close()
            # plt.show()
            # print(strategies)

            # plot approximation_error and save plot
            ae_png_filename = 'approximation_error-' + filename + '.png'
            ae_lists = [dict[sample_strategy.__name__]["avg_ks_dist"] for sample_strategy in strategies]
            ae_lists = bring_lists_to_same_length(ae_lists)
            x = range(len(ae_lists[0]))
            plt.xlabel('# samples')
            plt.ylabel('true approximation error')
            plt.title('averages for ' + filename)
            for i in range(len(ae_lists)):
                plt.plot(x, ae_lists[i], label=strategies[i].__name__)
            plt.plot(x, [rho]*len(ae_lists[0]), label='rho')
            plt.legend()
            plt.savefig(ae_png_filename)
            plt.close()
            # plt.show()

