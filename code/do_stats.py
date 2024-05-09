# compute domain similarity.
import pandas as pd
import numpy as np

def compute(df, filename, dataset):
    number_of_users = df['user'].nunique()
    number_of_items = df['item'].nunique()
    number_of_interactions = len(df)
    df['text_length'] = df['explanation'].apply(lambda x: len(x.split()))
    average_length = df['text_length'].mean()
    density = number_of_interactions / (number_of_users * number_of_items) * 100
   
    with open(filename, 'a') as stats_file:
        stats_file.write(f"Dataset: {dataset}\n")
        stats_file.write(f"Number of users: {number_of_users}\n")
        stats_file.write(f"Number of items: {number_of_items}\n")
        stats_file.write(f"Number of interactions: {number_of_interactions}\n")
        stats_file.write(f"Average tokens per explanation: {average_length}\n")
        stats_file.write(f"Density: {density}\n")
        stats_file.write("\n") 
        stats_file.write("\n")
        stats_file.write("\n")


if __name__ == "__main__":
    datasets = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]
    for i, dataset in enumerate(datasets):
        df = pd.read_csv(f"./data/{dataset}/processed.csv")
        compute(df, "stats.out", dataset)
