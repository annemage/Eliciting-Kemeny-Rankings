import math

import numpy as np
from numpy import unravel_index
import itertools
import ortools
from ortools.linear_solver import pywraplp
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy


np.random.seed(24)
###########################################
######       BUILDING PROFILES       ######
###########################################

def generate_profile(n, k):  # CHECKED - CORRECT
    # generate a profile with n voters and k candidates uniformly at random
    profile = []
    for v in range(n):
        order = np.arange(k)
        np.random.shuffle(order)
        profile.append(order)
    return profile

def geometric_series(a, m):
    gs = 0
    for i in range(m):
        gs += a**i
    return gs


def generate_profile_Mallows(n, k, phi):  #
    # generate a profile with n voters and k candidates drawn from Mallows phi model,
    # because of symmetry, the reference ranking is always assumed to be 0,1,2,3,...,k
    # we use a repeated insertion model as described in
    # https://www.researchgate.net/publication/221345217_Learning_Mallows_Models_with_Pairwise_Preferences
    # here, we first generate a RIM vector based on some probabilities and then repeatedly insert elements from the
    # reference ranking into the position indicated by the RIM vector

    profile = []

    for _ in range(n):              # for all voters
        order = []
        for i in range(1, k+1):     # for all positions in the RIM vector
            probabilities = [phi ** (i - j) / geometric_series(phi, i) for j in range(1,i+1)]
                                    # probabilities_ij = probability that i-th entry of RIM vector is j. always, j <= i
            # print(probabilities)
            position = np.random.choice(i, p=probabilities) # here we draw the position in which i-1 is inserted
            order.insert(position, i-1)
        profile.append(order)
    return profile

def build_matrix_from_profile(profile):  # CHECKED - CORRECT
    n = len(profile)
    k = len(profile[0])
    Q = np.zeros((k,k))
    for v in range(n):
        for i in range(k):
            for j in range(i+1,k):
                a = int(profile[v][i])
                b = int(profile[v][j])
                Q[a][b] += 1
    Q = Q/n
    for i in range(k):
        Q[i][i] = 0.5
    return Q

def generate_sp_vote_profile_uar(n, m):     # generate a single-peaked profile on axis 0, ... , m-1
                                            # with n voters and m candidates uniformly at random
    P = np.empty((0, m), int)
    for i in range(n):                      # for all voters ...
        vote = np.empty(0, int)
        left = 0                            # assign left and right most of unassigned candidates
        right = m-1
        while left != right:                # build up ranking sequentially from the bottom uar
            if np.random.randint(2) == 0:   # with probability 1/2 append with the left most unassigned candidate
                vote = np.append(vote, left)
                left = left + 1
            else:                           # otherwise append with the left most unassigned candidate
                vote = np.append(vote, right)
                right = right - 1
        vote = np.append(vote, left)        # if left- and right-most candidate coincide insert this last candidate
        vote = np.flip(vote)                # flip the order of candidates around (start with the top ranked candidate)
        P = np.append(P, [vote], axis=0)    # add to the profile
    return P

# Under construction!!!!
def learn_sp_axis(P, gamma):    # recover the single-peaked axis of profile P when voters are queried one by one and
                                # with probability gamma abreviate questioning after every question round
    n = len(P)
    m = len(P[0])
    possible_successors = np.empty((0, n), int)
    for i in range(n):
        possible_successors = np.append(possible_successors, np.arange(n), axis=0)
    total_length = sum(len(row) for row in possible_successors)
    current_voter = 0
    while total_length > n-1:     # while not every voter has exactly one successor except for one that has 0 successors
        while np.random.binomial(1, gamma) == 0:
            already_asked = np.empty((0, 2), int)
            # ask current_voter a new question
            # append already_asked question
            # make inference about sp axis:
            #   if i>j, k>j then 1. j < i < k for any x with i > x > k or k > x > i
        total_length = sum(len(row) for row in possible_successors)
        current_voter = current_voter + 1
    axis = np.empty(0, int)
    return axis

def build_majority_graph(P):
    n = len(P)
    m = len(P[0])
    Q = np.zeros((m, m))
    for i in range(n):
        for c1 in range(m):
            for c2 in range(c1+1, m):
                winner = P[i][c1]
                defeated = P[i][c2]
                Q[winner][defeated] += 1
    Q = Q/n + 0.5 * np.eye(m)
    return Q
