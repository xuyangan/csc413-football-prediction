import glob
import datetime
import pandas as pd
import csv
import torch as th


def convert_to_datasets():
    # %%
    # read from xsls file

    file_pattern = 'berrar_ratings/data_recent_and_val_*.csv'
    files = glob.glob(file_pattern)

    full_df = pd.DataFrame()

    for file in files:
        df = pd.read_csv(file)
        # change the format of the Date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        full_df = pd.concat([full_df, df])

    # drop the Sea, Lge, GD, WDL columns
    full_df = full_df.drop(columns=['Sea', 'Lge', 'GD', 'WDL'])


    # %%
    # convert the teams to index
    teams = full_df['HT'].unique()
    teams.sort()

    # create a dictionary to map the team names to integers
    team_to_idx = {team: idx for idx, team in enumerate(teams)}
    idx_to_team = {idx: team for team, idx in team_to_idx.items()}

    # add the team index to the dataframe
    full_df['HT'] = full_df['HT'].map(team_to_idx)
    full_df['AT'] = full_df['AT'].map(team_to_idx)

    # %%
    # sort the dataframe by date
    full_df = full_df.sort_values(by='Date')

    split_date = "14/04/2023"
    split_date = datetime.datetime.strptime(split_date, "%d/%m/%Y")

    df_train = full_df[full_df['Date'] < split_date]
    df_val = full_df[full_df['Date'] >= split_date]

    # %%
    # convert the dataframe into pytorch tensors
    training_teams = th.tensor(df_train[['HT', 'AT']].values, dtype=th.int64)
    training_ratings = th.tensor(df_train[['HT_H_Off_Rating', 
                                            'HT_H_Def_Rating',
                                            'HT_A_Off_Rating', 
                                            'HT_A_Def_Rating',
                                            'AT_H_Off_Rating', 
                                            'AT_H_Def_Rating',
                                            'AT_A_Off_Rating',
                                            'AT_A_Def_Rating']].values, dtype=th.float32)

    training_data = th.cat([training_teams, training_ratings], dim=1)
    training_labels = th.tensor(df_train[['HS', 'AS']].values, dtype=th.float32)

    testing_teams = th.tensor(df_val[['HT', 'AT']].values, dtype=th.int64)
    testing_ratings = th.tensor(df_val[['HT_H_Off_Rating', 
                                        'HT_H_Def_Rating',
                                        'HT_A_Off_Rating', 
                                        'HT_A_Def_Rating',
                                        'AT_H_Off_Rating', 
                                        'AT_H_Def_Rating',
                                        'AT_A_Off_Rating',
                                        'AT_A_Def_Rating']].values, dtype=th.float32)
    testing_data = th.cat([testing_teams, testing_ratings], dim=1)
    testing_labels = th.tensor(df_val[['HS', 'AS']].values, dtype=th.float32)

    # separate the training set into training and validation sets
    split = int(0.8 * training_data.shape[0])
    train_data = training_data[:split]
    train_labels = training_labels[:split]
    val_data = training_data[split:]
    val_labels = training_labels[split:]


    return train_data, train_labels, val_data, val_labels, testing_data, testing_labels, team_to_idx, idx_to_team