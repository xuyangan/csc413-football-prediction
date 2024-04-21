# %%
import pandas as pd
import glob
import numpy as np

# %%
features_to_keep = [    
                        "HT",
                        "AT",
                        "HS",
                        "AS",
                        "HST",
                        "AST",
                        "HC",
                        "AC",
                        "HF",
                        "AF",
                        "HFKC",
                        "AFKC",
                        "HY",
                        "AY",
                        "HR",
                        "AR",
                        "B365H",
                        "B365D",
                        "B365A",
                        'HT_H_Off_Rating', 
                        'HT_H_Def_Rating',
                        'HT_A_Off_Rating', 
                        'HT_A_Def_Rating',
                        'AT_H_Off_Rating', 
                        'AT_H_Def_Rating',
                        'AT_A_Off_Rating',
                        'AT_A_Def_Rating'
]

labels_to_keep = [  
                    "HT",
                    "AT",
                    "FTR",
                    "FTHG",
                    "FTAG"
]


# %%
def load_berrar_ratings():
    file_pattern = 'berrar_ratings/full*.csv'
    files = glob.glob(file_pattern)
    full_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        full_df = pd.concat([full_df, df])
    full_df = full_df.reset_index().set_index('Date')
    full_df = full_df.sort_values('Date')
    full_target = full_df[labels_to_keep]
    full_data = full_df[features_to_keep]
    return full_df, full_data, full_target

# %%
full_df, full_data, full_target = load_berrar_ratings()
full_data

# %%
# test how well the berrar ratings do for predicting A vs H
def test_berrar_ratings(df):
    total = 0
    right_counts = 0
    # keep "FTR", "HT_EG", "AT_EG"
    small = df[["FTR", "HT_EG", "AT_EG"]]
    # change to list of lists
    for row in small.values:
        pred_home_goals = row[1]
        pred_away_goals = row[2]

        if pred_home_goals > pred_away_goals:
            pred = "H"
        elif pred_home_goals < pred_away_goals:
            pred = "A"
        if pred == row[0]:
            right_counts += 1
        if row[0] == "H" or row[0] == "A":
            total += 1
    print("Berrar ratings accuracy for only A and H: ", right_counts/total)
    return None

test_berrar_ratings(full_df)

# %%
home_team_features = [
                        'HT_H_Off_Rating',
                        'HT_H_Def_Rating',
                        'HT_A_Off_Rating',
                        'HT_A_Def_Rating',
                        "HS",
                        "HST",
                        "HC",
                        "HF",
                        "HFKC",
                        "HY",
                        "HR",
                        "B365H",
                        "B365D",
                        "B365A"
]   

away_team_features = [  
                        'AT_H_Off_Rating',
                        'AT_H_Def_Rating',
                        'AT_A_Off_Rating',
                        'AT_A_Def_Rating',
                        "AS",
                        "AST",
                        "AC",
                        "AF",
                        "AFKC",
                        "AY",
                        "AR",
                        "B365H",
                        "B365D",
                        "B365A"
]


# %%
def get_team_history_rating(df, label, team, n_stagger, before_date, features):
        before_df = df.reset_index().set_index(["Date"]).sort_index()
        # before_df = before_df.loc[:before_date]
        # get the games before the date or equal to the date
        before_df = before_df[before_df.index <= before_date]
        before_df = before_df[before_df[label] == team]
        # if there is less than n_stagger games before the date, 
        eval = n_stagger - len(before_df)
        if eval > 0:
            return None 
            # nan_rows = pd.DataFrame(np.nan, index=range(eval), columns=before_df.columns)
            # before_df = pd.concat([nan_rows, before_df])
            # before_df = before_df.fillna(0)
        
        else:
            before_df = before_df.iloc[-n_stagger:]
        before_df = before_df[features]
        return before_df

        
# test
# get_team_history_rating(full_data, 'AT', 'Liverpool', 5, '2019-12-26', away_team_features)

# %%

result_map = {  
            "H": 0,
            "D": 1,
            "A": 2
}

teams = full_df["HT"].unique()
teams_to_idx = {team: idx for idx, team in enumerate(teams)}
idx_to_teams = {idx: team for idx, team in enumerate(teams)}
# change the team names to indexes
full_data["HT"] = full_data["HT"].map(teams_to_idx)
full_data["AT"] = full_data["AT"].map(teams_to_idx)
full_target["HT"] = full_target["HT"].map(teams_to_idx)
full_target["AT"] = full_target["AT"].map(teams_to_idx)


# %%
import torch as th

def compute_history(full_data, full_target, n_stagger):

    data_dp = full_data.copy()
    target_dp = full_target.copy().reset_index()
    data_dp.reset_index().set_index(["Date", "HT", "AT"]).sort_index()
    target_dp.reset_index().set_index(["Date", "HT", "AT"]).sort_index()


    # shape num_matches x n_stagger x num_features
    matches_features_home = []
    matches_features_away = []

    targets = []

    # map the result to a number
    # target_dp = target_dp["FTR"].map(result_map)
    # targets = th.tensor(target_dp.values, dtype=th.long)

    home_teams_matches = []
    away_teams_matches = []

    for index, row in data_dp.iterrows():
        # this is match
        date = index
        home_team = row["HT"]
        away_team = row["AT"]

        # there are n_stagger rows with num_features columns
        home_history = get_team_history_rating(data_dp, 'HT', home_team, n_stagger, date, home_team_features)
        if home_history is None:
            continue
        # shape n_stagger x num_features
        away_history = get_team_history_rating(data_dp, 'AT', away_team, n_stagger, date, away_team_features)
        if away_history is None:
            continue
        
        # shape n_stagger x num_features
        feature_home = th.tensor(home_history.values, dtype=th.float)
        feature_away = th.tensor(away_history.values, dtype=th.float)

        # append in another dimension
        matches_features_home.append(feature_home.unsqueeze(0))
        matches_features_away.append(feature_away.unsqueeze(0))

        target = target_dp.loc[target_dp["Date"] == date]
        target = target.loc[target["HT"] == home_team]
        target = target.loc[target["AT"] == away_team]
        target = target["FTR"].values[0]
        targets.append(result_map[target])

        home_teams_matches.append(home_team)
        away_teams_matches.append(away_team)
        
    matches_features_home = th.cat(matches_features_home, dim=0)
    matches_features_away = th.cat(matches_features_away, dim=0)
    targets = th.tensor(targets, dtype=th.long)
    home_teams_matches = th.tensor(home_teams_matches, dtype=th.long)
    away_teams_matches = th.tensor(away_teams_matches, dtype=th.long)

    return home_teams_matches, away_teams_matches, matches_features_home, matches_features_away, targets


# %%
# separate into training and test set

date = "2023-04-14"

training_data_df = full_data.loc[:date]
training_target_df = full_target.loc[:date]

test_data_df = full_data.loc[date:]
test_target_df = full_target.loc[date:]


training_home_teams_matches, training_away_teams_matches, \
    training_matches_features_home, training_matches_features_away, \
        training_targets = compute_history(training_data_df, training_target_df, 5)

test_home_teams_matches, test_away_teams_matches, \
    test_matches_features_home, test_matches_features_away, \
        test_targets = compute_history(test_data_df, test_target_df, 5)



# %%
print(training_home_teams_matches.shape)
print(training_away_teams_matches.shape)
print(training_matches_features_away.shape)
print(training_matches_features_home.shape)
print(training_targets.shape)
print("-------------------")
print(test_home_teams_matches.shape)
print(test_away_teams_matches.shape)
print(test_matches_features_home.shape)
print(test_matches_features_away.shape)
print(test_targets.shape)

# %%
import pickle
# save the tensors to disk


folder = "tensors"
th.save(training_home_teams_matches, f'{folder}/training_home_teams_matches.pt')
th.save(training_away_teams_matches, f'{folder}/training_away_teams_matches.pt')
th.save(training_matches_features_home, f'{folder}/training_matches_features_home.pt')
th.save(training_matches_features_away, f'{folder}/training_matches_features_away.pt')
th.save(training_targets, f'{folder}/training_targets.pt')

th.save(test_home_teams_matches, f'{folder}/test_home_teams_matches.pt')
th.save(test_away_teams_matches, f'{folder}/test_away_teams_matches.pt')
th.save(test_matches_features_home, f'{folder}/test_matches_features_home.pt')
th.save(test_matches_features_away, f'{folder}/test_matches_features_away.pt')
th.save(test_targets, f'{folder}/test_targets.pt')


# save the dictionaries

with open(f'{folder}/result_map.pkl', 'wb') as f:
    pickle.dump(result_map, f)

with open(f'{folder}/teams_to_idx.pkl', 'wb') as f:
    pickle.dump(teams_to_idx, f)

with open(f'{folder}/idx_to_teams.pkl', 'wb') as f:
    pickle.dump(idx_to_teams, f)


# %%

