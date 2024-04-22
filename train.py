from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torch


def one_hot_targets(targets):
    num_classes = torch.max(targets).item() + 1
    return F.one_hot(targets, num_classes=num_classes).type(torch.float32)

def get_test_data():
    test_home_teams_matches = torch.load("dataset/tensors/test_home_teams_matches.pt")
    test_away_teams_matches = torch.load("dataset/tensors/test_away_teams_matches.pt")
    test_matches_features_home = torch.load("dataset/tensors/test_matches_features_home.pt")
    test_matches_features_away = torch.load("dataset/tensors/test_matches_features_away.pt")
    test_targets = torch.load("dataset/tensors/test_targets.pt")

    test_targets_oh = one_hot_targets(test_targets)

    test_dataset = TensorDataset(test_home_teams_matches, test_away_teams_matches, test_matches_features_home,
                                test_matches_features_away, test_targets_oh)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return test_dataset

def get_train_val_data(split=0.8):
    training_home_teams_matches = torch.load("dataset/tensors/training_home_teams_matches.pt")
    training_away_teams_matches = torch.load("dataset/tensors/training_away_teams_matches.pt")
    training_matches_features_home = torch.load("dataset/tensors/training_matches_features_home.pt")
    training_matches_features_away = torch.load("dataset/tensors/training_matches_features_away.pt")
    training_targets = torch.load("dataset/tensors/training_targets.pt")

    training_targets_oh = one_hot_targets(training_targets)

    split_idx = int(len(training_home_teams_matches) * split)

    train_dataset = TensorDataset(training_home_teams_matches[:split_idx], training_away_teams_matches[:split_idx],
                                training_matches_features_home[:split_idx], training_matches_features_away[:split_idx],
                                training_targets_oh[:split_idx])
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # true could cause lookahead

    val_dataset = TensorDataset(training_home_teams_matches[split_idx:], training_away_teams_matches[split_idx:],
                                training_matches_features_home[split_idx:], training_matches_features_away[split_idx:],
                                training_targets_oh[split_idx:])
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_dataset, val_dataset
