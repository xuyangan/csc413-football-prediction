# %%
import pandas as pd
import torch as th
import torch.nn as nn
import glob
import pickle

import sys
sys.path.append('../models/')
from models.inception1D import *

# %%

training_home_teams_matches = th.load("dataset/tensors/training_home_teams_matches.pt")
training_away_teams_matches = th.load("dataset/tensors/training_away_teams_matches.pt")
training_matches_features_home = th.load("dataset/tensors/training_matches_features_home.pt")
training_matches_features_away = th.load("dataset/tensors/training_matches_features_away.pt")
training_targets = th.load("dataset/tensors/training_targets.pt")

test_home_teams_matches = th.load("dataset/tensors/test_home_teams_matches.pt")
test_away_teams_matches = th.load("dataset/tensors/test_away_teams_matches.pt")
test_matches_features_home = th.load("dataset/tensors/test_matches_features_home.pt")
test_matches_features_away = th.load("dataset/tensors/test_matches_features_away.pt")
test_targets = th.load("dataset/tensors/test_targets.pt")



# %%
import pickle
file1 = 'dataset/tensors/idx_to_teams.pkl'
file2 = 'dataset/tensors/teams_to_idx.pkl'
file3 = 'dataset/tensors/result_map.pkl'

with open(file1, 'rb') as file:
    idx_to_teams = pickle.load(file)

with open(file2, 'rb') as file:
    teams_to_idx = pickle.load(file)

with open(file3, 'rb') as file:
    result_map = pickle.load(file)

print(result_map)
idx_to_result = {0: 'H', 1: 'D', 2: 'A'}



# %%
print("Training data")
print(training_home_teams_matches.shape) # the idex of home team for the matches
print(training_away_teams_matches.shape) # the idex of away team for the matches
print(training_matches_features_home.shape) # the features of the home team for the matches
print(training_matches_features_away.shape) # the features of the away team for the matches
print(training_targets.shape) # the targets of the matches H D A

print("Test data")
print(test_home_teams_matches.shape)
print(test_away_teams_matches.shape)
print(test_matches_features_home.shape)
print(test_matches_features_away.shape)
print(test_targets.shape)




# %%
index = 0
print("First match")
print("Home team: ", idx_to_teams[training_home_teams_matches[index].item()])
print("Away team: ", idx_to_teams[training_away_teams_matches[index].item()])
print("Features home: ", training_matches_features_home[index])
print("Features away: ", training_matches_features_away[index])
print("Target: ", idx_to_result[training_targets[index].item()])


# %%
#split the data into training and validation
from torch.utils.data import DataLoader, TensorDataset

split = 0.8
split_idx = int(len(training_home_teams_matches) * split)

train_dataset = TensorDataset(training_home_teams_matches[:split_idx], training_away_teams_matches[:split_idx], training_matches_features_home[:split_idx], training_matches_features_away[:split_idx], training_targets[:split_idx])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(training_home_teams_matches[split_idx:], training_away_teams_matches[split_idx:], training_matches_features_home[split_idx:], training_matches_features_away[split_idx:], training_targets[split_idx:])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(test_home_teams_matches, test_away_teams_matches, test_matches_features_home, test_matches_features_away, test_targets)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Training data")
print(len(train_dataset))
print(len(train_loader))

print("Validation data")
print(len(val_dataset))
print(len(val_loader))

print("Test data")
print(len(test_dataset))
print(len(test_loader))

# %%
for home_teams, away_teams, home_features, away_features, targets in train_loader:
    print(home_teams.shape)
    print(away_teams.shape)
    print(home_features.shape)
    print(away_features.shape)
    print(targets.shape)
    break


