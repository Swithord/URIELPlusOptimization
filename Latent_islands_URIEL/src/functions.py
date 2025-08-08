import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import datasets, transforms as T
from torchsummary import summary
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mutual_info_score
from scipy.special import logsumexp

import networkx as nx
from graphviz import Digraph

import pickle

# input: data
# output: the mutual information matrix
def MI(data):
  features = data.shape[1]
  MI_matrix = np.zeros((features, features))
  for i in range(features):
    for j in range(features):
      if i != j: # want the diagonals to be zero so that variables aren't grouped with itself
        MI_matrix[i, j] = mutual_info_score(data[:, i], data[:, j])
  return MI_matrix

# calculate BIC values
def BIC(model, n_samples):
  ll = model["log_likelihood"]
  # print("log likelihood", ll)
  k = model["n_params"]
  scalar = 2
  # return k*np.log(n_samples) - 2*ll
  return scalar*k**2*np.log(n_samples)-2*ll
  # The modified BIC value ensures islands aren't too large

# epectation maximisation
def expectation_maximisation_LCM(D, n_states, n_iterations = 150, delta = 1e-3, verbose = False):
  # n_states is number of states the latent variable could be
  n_samples, n_vars = D.shape
  categories = 2 # uriel is binary so there are two categories

  # states is number of latent variables for each island
  P_z = np.full(n_states, 1.0 / n_states) # prior distribution assumed to be uniform

  P_x_given_z = [np.random.dirichlet(np.ones(categories), size=n_states) for k in range(n_vars)]
  # for each of the n_states there is one corresponding array
  # for each state, there is a non-negative array of length 2 that sums to 1
  # print("conditional probability: ", P_x_given_z)

  prev_ll = -np.inf
  if verbose:
    print(f"\n Current number states = {n_states}")
  for iter in range(n_iterations):
    if verbose and iter % 5 == 0:
      print("iteration: ", iter)
    # calculating expectation
    # log_probs = np.log(P_z)
    log_probs = np.tile(np.log(P_z), (n_samples, 1))

    for i in range(n_vars):
      x = D[:,i] # column i
      probs = P_x_given_z[i][:, x].T
      log_probs += np.log(probs + 1e-10) # in case it's zero

    log_post = log_probs - logsumexp(log_probs, axis = 1, keepdims = True)
    post = np.exp(log_post)

    # maximisation
    P_z = post.mean(axis = 0)

    for i in range(n_vars):
      x = D[:, i]
      k = categories # just 2
      P_x_given_z[i] = np.zeros((n_states, k))
      for j in range(k):
        mask = (x == j)
        P_x_given_z[i][:, j] = post[mask].sum(axis = 0)
        P_x_given_z[i][:, j] += 1e-10 # prevents division by 0 error in next line
      P_x_given_z[i] /= P_x_given_z[i].sum(axis = 1, keepdims = True)

    if np.abs(prev_ll - np.sum(logsumexp(log_probs, axis = 1))) < delta:
      break
    prev_ll = np.sum(logsumexp(log_probs, axis = 1))
    log_likelihood = prev_ll
    n_params = n_states * n_vars
  currModel = {"log_likelihood": log_likelihood,
               "n_params": n_params}
  currBic = BIC(currModel, n_samples)
  return {"P_z": P_z,
          "P_x_given_z": P_x_given_z,
          "log_likelihood": log_likelihood,
          "n_states": n_states,
          "n_params": n_params,
          "posterior": post,
          "increased_nodes": False,
          "bic": currBic,
          "n_vars": n_vars
          }
# learn a latent class model
def learnLCM(D, max_states = 2, verbose = False, n_restarts = 5):
  n_samples, n_vars = D.shape
  m = expectation_maximisation_LCM(D, 2, verbose = verbose)
  bic = BIC(m, n_samples = D.shape[0])

  for i in range(2, max_states + 1):
    for _ in range(n_restarts):
      m1 = expectation_maximisation_LCM(D, i, verbose = verbose)
      bic1 = BIC(m1, n_samples = n_samples)
      if bic1 <= bic:
        m = m1
        bic = bic1
  return m

def twolayerBIC(latent1, latent2, n_samples):
  # ll = latent1["log_likelihood"] + latent2["log_likelihood"]
  # k = latent1["n_params"] + latent2["n_params"]
  # k = max(latent1["n_params"], latent2["n_params"])
  bic1 = BIC(latent1, n_samples = n_samples)
  bic2 = BIC(latent2, n_samples = n_samples)
  return bic1 + bic2

# node introduction
def NI(D, model,mutual_info):
  n_samples, n_vars = D.shape
  best = None
  best_bic = np.inf
  bestgroup1, bestgroup2 = None, None
  joint = None
  P_Z2_given_Z1 = None

  vars = list(range(D.shape[1]))
  # all_pairs = list(combinations(vars, 2))

  k = 3 # number of pairs to consider
  pairs = [(mutual_info[i,j],i,j) for i in range(n_vars) for j in range(i+1, n_vars)]
  pairs = sorted(pairs, key=lambda x: x[0], reverse = True)[:k]

  # print("bic is: ", best_bic)
  for (_,i, j) in pairs:
    # get pair that will be placed in own branch
    Z_group_new = [i,j]
    Z_group = [k for k in vars if k not in Z_group_new]

    # get the data for each group
    D_Z_new = D[:, Z_group_new]
    D_Z = D[:, Z_group]

    m_new = learnLCM(D_Z_new)
    m = learnLCM(D_Z)

    # calculate new bic value
    bic = twolayerBIC(m, m_new, n_samples = D.shape[0])
    # print(f"bic({i},{j}) is: {bic:.3f}")

    # calculate joint distribution
    # joint = np.dot(m["posterior"].T, m_new["posterior"])
    # P_Z2_given_Z1 = joint / joint.sum(axis = 1, keepdims = True)

    if bic <= best_bic:
      best = (m, m_new)
      best_bic = bic
      bestgroup1, bestgroup2 = Z_group, Z_group_new
      # calculate joint distribution
      joint = np.dot(m["posterior"].T, m_new["posterior"])
      P_Z2_given_Z1 = joint / joint.sum(axis = 1, keepdims = True)
  return {
      "latent": ["Z1", "Z2"],
      "group1": bestgroup1,
      "group2": bestgroup2,
      "Z1": best[0],
      "Z2": best[1],
      "P_Z2_given_Z1": P_Z2_given_Z1,
      "bic": best_bic,
      "increased_nodes": True,
      "structure": {
          "Z1": bestgroup1 + ["Z2"],
          "Z2": bestgroup2
        }
      }

def NR(model, group1, group2, D):
  n_samples, n_vars = D.shape
  best = model
  best_bic = model["bic"]
  G1, G2 = group1, group2
  for element in group1:
    # group 1 is the group higher in the initial root
    new_group1 = [x for x in G1 if x != element]
    # group 2 is the group associated with the second latent node
    new_group2 = G2 + [element]

    # if group1 is empty there's nothing to work with
    if not new_group1:
      continue

    D_Z1 = D[:, new_group1]
    D_Z2 = D[:, new_group2]

    # m1 = expectation_maximisation_LCM(D_Z1, 2)
    # m2 = expectation_maximisation_LCM(D_Z2, 2)
    m1 = learnLCM(D_Z1)
    m2 = learnLCM(D_Z2)

    joint = np.dot(m1["posterior"].T, m2["posterior"])
    P_Z2_given_Z1 = joint / joint.sum(axis = 1, keepdims = True)

    bic = twolayerBIC(m1, m2, n_samples = n_samples)
    # print("group1: ", new_group1, "group2: ", new_group2)
    # print(bic)

    if bic < best_bic:
      best_bic = bic
      best = {
        "latent": ["Z1", "Z2"],
        "group1": new_group1,
        "group2": new_group2,
        "Z1": m1,
        "Z2": m2,
        "P_Z2_given_Z1": P_Z2_given_Z1,
        "bic": best_bic,
        "increased_nodes": True,
        "structure": {
            "Z1": new_group1,
            "Z2": new_group2
          }
        }
      G1, G2 = new_group1, new_group2
  print("NR: G1: ", G1, "G2: ", G2, "oldbic: ", model["bic"], "newbic: ", best_bic)
  return best

def learnLTM(D, mutual_info, model, max_states = 2):
  n_samples, n_vars = D.shape
  # model = expectation_maximisation_LCM(D, n_states = 2) # start with a 2-state latent variable model

  # use learnLCM before function call and pass it here using model term
  # model = learnLCM(D, max_states)
  best = model
  best_bic = model["bic"]

  model_candidate = NI(D, model, mutual_info)

  # print("increased_nodes: ", model_candidate["increased_nodes"])
  if model_candidate["increased_nodes"]:
    # no need to run SI since NR does that already
    # model_candidate = SI_two_layer(D, model)
    model_candidate = NR(model_candidate, model_candidate["group1"], model_candidate["group2"], D)

  print("(in LTM) The two bics are (NR vs LCA): ", model_candidate["bic"], best_bic)
  if model_candidate["bic"] <= best_bic:
    best_bic = model_candidate["bic"]
    best = model_candidate
  return best

# Bridged island algorithm

# D is data
# V is the set of attributes (features for URIEL)
# delta is the thresholding number that can be used for tuning
def bridged_islands(D, delta = 0.001, mutual_info = None):
  n_samples, n_vars = D.shape
  islands = []
  V = set(range(n_vars))
  # V will be a set of indices that diminishes as they are assigned islands
  if mutual_info is None:
    mutual_info = MI(D)

  while V:
    print(len(V))
    # S = pair of variables with highest MI
    # mutual_info = MI(D)
    if len(V) == 1:
      tempModel = learnLCM(D[:, list(V)])
      temp_original = [sorted(list(V))[i] for i in range(tempModel["n_vars"])]
      islands.append({"group_projected" : [i for i in range(len(V))], "indices": list(V), "model": tempModel})
      V.remove(list(V)[0])
      break

    x,y = -1,-1
    best_mutual_info = -np.inf
    for i in V:
      for j in V:
        if i != j and mutual_info[i,j] > best_mutual_info:
          best_mutual_info = mutual_info[i,j]
          x, y = i, j
    S = set([x,y])
    if x == -1 or y == -1:
      print("the first if statement did not catch it")

    # make an island
    while True:
      print("S: ", S, "V: ", V)
      # print("islands: ", [islands[i]["indices"] for i in range(len(islands))])
      # S = set([x,y])

      # candidates = V-set(S) # potential variables to add to the same class
      V -= set(S) # potential variables to add to the same class
      selected = -1
      highest_MI = -np.inf
      # for c in candidates:
      for c in V:
        currTotal = 0
        for s in S:
          currTotal += mutual_info[s,c]
        if currTotal > highest_MI:
          highest_MI = currTotal
          selected = c

      if selected == -1:
        tempModel = learnLCM(D[:, list(S)])
        temp_original = [sorted(list(S))[i] for i in range(tempModel["n_vars"])]
        islands.append({"group_projected" : [i for i in range(len(S))], "indices": list(S), "model": tempModel})
        # print("islands remaining: ", [islands[i]["indices"] for i in range(len(islands))])
        break
      S.add(selected)
      V.remove(selected)
      # print("S: ", S)
      # print("V: ", V)

      sorted_S = sorted(S)
      D1 = D[:, sorted_S]
      # print("current D1 = ", D1)

      m1 = learnLCM(D1)

      mutualInfo = mutual_info[np.ix_(sorted_S, sorted_S)]
      m2 = learnLTM(D1, mutualInfo, m1)

      print("bic m1: ", m1["bic"], "bic m2: ", m2["bic"], m2["increased_nodes"])
      # print("changed: ", m2["increased_nodes"])
      if m2["increased_nodes"]:
        print("m2bic/m1(class)bic: ", m2["bic"]/m1["bic"])
        if m2["bic"] < m1["bic"] + delta:
          print("tree/class ratio: ", m2["bic"]/m1["bic"])
        # if m2["bic"]/m1["bic"] > 0.97:
          G1 = m2["group1"]
          G1_original = [sorted(list(S))[i] for i in G1]
          G2 = m2["group2"]
          G2_original = [sorted(list(S))[i] for i in G2]
          # larger group will form island
          if len(G2) > len(G1):
            islands.append({"group_projected" :G2, "indices": G2_original, "model": m2["Z2"]})
            print("added G2: ", G2_original)
            V |= set(G1_original)
            break
          else:
            islands.append({"group_projected" :G1, "indices": G1_original, "model": m2["Z1"]})
            print("added G1: ", G1_original)
            V |= set(G2_original)
            break
          # print("original added ", G1_original, G2_original)
  # print("remaining; ", V)
  return islands


def prob_latent(model, observation):
  P_z = model["P_z"]
  P_x_given_z = model["P_x_given_z"]

  log_prob_z = np.log(P_z)
  # log p(z|obs) similar to log(P(z)) + sum_i log P(obs_i | z)
  for index, obs in enumerate(observation):
    log_prob_z += np.log(P_x_given_z[index][:, obs])
  # normalisation
  log_prob_z -= logsumexp(log_prob_z)
  return np.exp(log_prob_z)[1]

def dist(model, observation1, observation2):
  # return np.linalg.norm(prob_latent(model, observation1) - prob_latent(model, observation2))
  x = np.array(observation1)
  y = np.array(observation2)
  magnitude = np.linalg.norm(x) * np.linalg.norm(y)
  if magnitude < 1e-10:
    return 0
  distance = 1 - np.dot(x,y) / magnitude # 1 - cosine similarity
  # print("distance: ", distance)
  return distance

# using probabilities
def dist2(model, observation1, observation2):
  x = prob_latent(model, observation1)
  y = prob_latent(model, observation2)
  magnitude = np.linalg.norm(x) * np.linalg.norm(y)
  if magnitude < 1e-10:
    return 0
  distance = 1 - np.dot(x,y) / magnitude # 1 - cosine similarity
  # print("distance: ", distance)
  return distance

def shannon_jensen_divergence(p1, p2):
  # p1 and p2 are probability distributions
  # p1, p2 = p1 / np.sum(p1), p2 / np.sum(p2) # normalise
  m = (p1 + p2) / 2
  return np.sum(p1 * np.log(p1 / m)) + np.sum(p2 * np.log(p2 / m))

def total_dist(islands, observation1, observation2):
  total = 0
  
  # langrank has decimals so it is 1. instead of 1
  obs1 = observation1.astype(int)
  obs2 = observation2.astype(int)

  x = np.array([prob_latent(island["model"], obs1[island["indices"]]) for island in islands])
  y = np.array([prob_latent(island["model"], obs2[island["indices"]]) for island in islands])
  # diff = x-y
  # z = np.array([np.linalg.norm(obs1[island["indices"]] - obs2[island["indices"]], ord = 2) for island in islands])
  # return np.linalg.norm(z)
  # return shannon_jensen_divergence(x,y)
  # norm_island_distances = np.linalg.norm(z, ord = 1)/ len(islands)
  # #####
  # x = x -np.mean(x)
  # y = y - np.mean(y)
  # return np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))
  # ######
  # x,y = observation1, observation2
  magnitude = np.linalg.norm(x) * np.linalg.norm(y)
  if magnitude < 1e-10:
    return 0
  distance = 1 - np.dot(x,y) / magnitude # 1 - cosine similarity
  # distance = np.linalg.norm(z*(np.abs(diff+1e-1)), ord = 1)
  # print("distance: ", distance)
  # return np.sum(np.abs(x-y))/len(islands) # average distance between the two observations
  # print("distance: ", distance)
  return distance
  # return np.linalg.norm(x - y, ord = 3) # norm
  # return np.random.uniform(0, 1)
  
def lang_to_index(languages, lang):
  result = np.where(languages == lang)
  if len(result[0]) != 1:
    print(f"language {lang} does not exist or multiple languages with same name")
    return None
  # print(result)
  return result[0][0]

def distance(D, islands, languages, lang1, lang2):
  x = lang_to_index(languages, lang1)
  y = lang_to_index(languages, lang2)
  
  if x is None or y is None:
    print("one of the languages is invalid")
    return None
  return total_dist(islands, D[x], D[y])

def distance_vector_input(islands, lang1, lang2):
  x = np.array(lang1)
  y = np.array(lang2)
  
  if len(x) != len(y):
    print("feature length different")
    return None
  return total_dist(islands, x, y)
  

# testing
# test = np.array([
#     [1,0,1,0,0,0,0,0],
#     [1,1,0,0,0,0,0,0],
#     [1,1,1,0,0,0,0,0],
#     [1,1,0,0,0,0,0,0],
#     [1,1,1,0,0,0,0,0],
#     [1,0,1,0,0,0,0,0],
#     [1,1,1,0,0,0,0,0],
#     [0,0,0,0,1,0,0,0],
#     [0,0,0,1,0,0,0,0],
#     [0,0,0,1,1,0,0,0],
#     [0,0,0,1,1,0,0,0],
#     [0,0,0,0,0,0,1,1],
#     [0,0,0,0,0,0,0,1],
#     [0,0,0,0,0,0,1,1],
#     [0,0,0,0,0,0,0,1],
#     [0,0,0,0,0,0,1,1],
#     [0,0,0,0,0,0,0,1],
#     [0,0,0,0,0,0,1,0]
# ])


# model = bridged_islands(D=test)
# print("islands: ", len(model))
# print(model)