# %%
import pandas as pd
import glob

# %%
def load_berrar_ratings():
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

    return full_df

# %%
df = load_berrar_ratings()

# %%
df["Date"] = pd.to_datetime(df["Date"])

# %%
df= df.reset_index().set_index(["HT","Date","AT"]).sort_index()

# %%
feature_list = [
                "HT_H_Off_Rating",
                "HT_H_Def_Rating",
                "HT_A_Off_Rating",	
                "HT_A_Def_Rating",	
                "AT_H_Off_Rating",	
                "AT_H_Def_Rating",
                "AT_A_Off_Rating",	
                "AT_A_Def_Rating",
                "HS",
                "AS"]

# %%
full_data_df_X = df[feature_list]
full_data_df_X = full_data_df_X.reset_index().set_index(["Date", "AT"])
# full_data_df_X.shift(3)

# %%

def transform_dataframe(df, n_stagger):
    # Initialize the output DataFrame
    # Shift features 1-7 by 5 time steps (T-5)
    combined = df.copy()
    col_ordered = list(df.columns)
    for i in range(n_stagger):
        T_shift = i+1
        shifted = df.shift(T_shift)
        column_names= [f"{col} T-{T_shift}" for col in df]
        shifted.columns = column_names
        col_ordered = column_names + col_ordered 
        if T_shift == 1: 
            combined = pd.merge(df, shifted, left_index=True, right_index=True)
        else:
            combined = pd.merge(combined, shifted, left_index=True, right_index=True)
    return combined[col_ordered]
def create_seq_group_by(df, features_list,n_stagger):
    list_of_df = [] 
    for i in df.HT.unique(): 
        temp = df[df["HT"] ==i]
        features_only = temp[features_list]
        seq_df = transform_dataframe(features_only, n_stagger)
        seq_df["HT"] = i 
        seq_df = seq_df.reset_index().set_index(["Date","HT", "AT"])
        list_of_df.append(seq_df)
        # break
    return list_of_df

# %%
seq_full_data_list_x = create_seq_group_by(full_data_df_X,feature_list,5)


# %%
seq_full_data_list_x_cat  = pd.concat(seq_full_data_list_x, axis=0)
seq_full_data_list_x_cat


# %%
# zero the nan values
seq_full_data_list_x_cat = seq_full_data_list_x_cat.fillna(0)
seq_full_data_list_x_cat

# %%
def save_to_csv(df, filename):
    df.to_csv(filename, index=True)

# %%
filename = "dataset/seq_target_data.csv"
save_to_csv(seq_full_data_list_x_cat, filename)


