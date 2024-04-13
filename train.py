# %%
import pandas as pd
import torch as th
import torch.nn as nn
import glob

import sys
sys.path.append('../models/')
from models.inception import *

# %%
def load_data():
    # %%
    # read from xsls file

    file_pattern = 'dataset/berrar_ratings/data_recent_and_val_*.csv'
    files = glob.glob(file_pattern)

    full_df = pd.DataFrame()

    for file in files:
        df = pd.read_csv(file)
        # change the format of the Date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        full_df = pd.concat([full_df, df])

    # drop the Sea, Lge, GD, WDL columns
    full_df = full_df.drop(columns=['Sea', 'Lge', 'GD', 'WDL'])


    # convert the teams to index
    teams = full_df['HT'].unique()
    teams.sort()

    # create a dictionary to map the team names to integers
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}

    # add the team index to the dataframe
    full_df['HT'] = full_df['HT'].map(team_to_idx)
    full_df['AT'] = full_df['AT'].map(team_to_idx)

    # sort the dataframe by date
    full_df = full_df.sort_values(by='Date')

    split_date = "14/04/2023"
    split_date = datetime.datetime.strptime(split_date, "%d/%m/%Y")

    df_train = full_df[full_df['Date'] < split_date]
    df_test = full_df[full_df['Date'] >= split_date]

    training_berrar_expected_goals = th.tensor(df_train[['HT_EG', 'AT_EG']].values, dtype=th.float32)
    testing_berrar_expected_goals = th.tensor(df_test[['HT_EG', 'AT_EG']].values, dtype=th.float32)

    training_data = th.tensor(df_train.drop(columns=['Date', 'HS', 'AS', 'HT_EG', 'AT_EG']).values, dtype=th.float32)
    training_labels = th.tensor(df_train[['HS', 'AS']].values, dtype=th.float32)

    testing_data = th.tensor(df_test.drop(columns=['Date', 'HS', 'AS', 'HT_EG', 'AT_EG']).values, dtype=th.float32)
    testing_labels = th.tensor(df_test[['HS', 'AS']].values, dtype=th.float32)

    # separate the training set into training and validation sets
    split = int(0.8 * training_data.shape[0])
    train_data = training_data[:split]
    train_labels = training_labels[:split]
    val_data = training_data[split:]
    val_labels = training_labels[split:]


    return train_data, train_labels, val_data, val_labels, testing_data, testing_labels, team_to_idx, idx_to_team, training_berrar_expected_goals, testing_berrar_expected_goals

# %%
train_data, train_labels, val_data, val_labels, testing_data, testing_labels, team_to_idx, idx_to_team, training_berrar_expected_goals, testing_berrar_expected_goals = load_data() 


# %%

time_step = 28
num_teams = len(team_to_idx)
embedding_dim = 10
in_channels = 1
batch_size = 100

train_data = train_data.unfold(0, time_step, 2)
train_labels = train_labels[time_step-1:]

val_data = val_data.unfold(0, time_step, 1)
val_labels = val_labels[time_step-1:]

testing_data = testing_data.unfold(0, time_step, 1)
testing_labels = testing_labels[time_step-1:]

data_loader = th.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)




# %%
# create the model
embedding_layer = TeamEmbedding(num_teams, embedding_dim)   
inception_model = Inceptionv2(in_channels)

# # test the model with a batch of data and see the output shape

for data in data_loader:
    print(data.shape)
    x = embedding_layer(data)
    x = x.unsqueeze(1)
    x = inception_model(x)



