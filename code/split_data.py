import pandas as pd
from sklearn.model_selection import train_test_split

def split_func(df, random_seed=42):
    user_dict = {user: idx for idx, user in enumerate(df['user'].unique())}
    item_dict = {item: idx for idx, item in enumerate(df['item'].unique())}

    # Create new columns user_idx and item_idx using mapping dictionaries
    df['user_idx'] = df['user'].map(user_dict)
    df['item_idx'] = df['item'].map(item_dict)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    valid_df = valid_df[valid_df['user_idx'].isin(train_df['user_idx']) & valid_df['item_idx'].isin(train_df['item_idx'])]
    test_df = test_df[test_df['user_idx'].isin(train_df['user_idx']) & test_df['item_idx'].isin(train_df['item_idx'])]
    return train_df, valid_df, test_df


if __name__ == "__main__":
    datasets = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]
    for i, dataset in enumerate(datasets):
        df = pd.read_csv(f"./data/{dataset}/processed.csv")
        train_df, valid_df, test_df = split_func(df)
        print(f"{dataset}: train:{len(train_df)}, valid:{len(valid_df)}, test:{len(test_df)}")
        train_df.to_csv("./data/"+dataset+"/train.csv", index=False)
        valid_df.to_csv("./data/"+dataset+"/valid.csv", index=False)
        test_df.to_csv("./data/"+dataset+"/test.csv", index=False)