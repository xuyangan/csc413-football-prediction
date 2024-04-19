# %%
import pandas as pd

# %%
# read from xsls file

data_full = pd.read_excel('data/TrainingSet-FINAL.xlsx')
val_full = pd.read_excel('data/PredictionSet-FINAL.xlsx')
val_full = val_full.drop(columns=['ID'])


# %%
# get the recent 5 seasons only
data_recent = data_full[data_full['Sea'].isin(data_full["Sea"].unique()[-5:])]

# add the validation set to the training set of the recent 5 seasons
data_recent_and_val = pd.concat([data_recent, val_full])

# %%
# write back to csv 
data_recent_and_val.to_csv('data/data_recent_and_val.csv', index=False)
