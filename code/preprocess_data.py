import pandas as pd
import numpy as np
import pickle
# save to df and keep only at least 5 interaction. 

if __name__ == "__main__":
    datasets = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]
    for dataset in datasets:
        with open(f"./data/{dataset}/reviews.pickle", "rb") as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        if dataset in ["AM_Movies", "AM_Electronics", "AM_CDs"]:
            df = df[df["sentence"].notna()]
            df['explanation'] = df['sentence'].apply(lambda x: x[0][2])
            df.rename(columns={"text": "review"}, inplace=True)
            df.drop(["sentence"], axis=1, inplace=True)
            
            
        else:
            if dataset == "Yelp": # too large, split two parts. 
                half_size = len(df) // 2
                df = df.sample(n=half_size, random_state=42)  # Set random_state for reproducibility
            else:
                scale_size = int(len(df) * 0.9)
                df = df.sample(n=scale_size, random_state=42)  # Set random_state for reproducibility
            df['explanation'] = df['template'].apply(lambda x: x[2])
            df['review'] = df['explanation']
            # Dropping unnecessary columns and 'template'
            df.drop(['template', 'predicted'], axis=1, inplace=True)

        # filter to 5. 
        for i in range(30):
            user_interactions = df['user'].value_counts()
            item_interactions = df['item'].value_counts()
            filtered_users = user_interactions[user_interactions >= 5].index
            filtered_items = item_interactions[item_interactions >= 5].index
            filtered_df = df[df['user'].isin(filtered_users) & df['item'].isin(filtered_items)]
            df = filtered_df
            
        print(f"{dataset} filtered to {len(df)} rows.")
        df.to_csv(f"./data/{dataset}/processed.csv", index=False)

    for dataset in datasets:
        df = pd.read_csv(f"./data/{dataset}/processed.csv")
        df['explanation'] = df['explanation'].fillna("No explanation provided")
        df.to_csv(f"./data/{dataset}/processed.csv", index=False)



        
            