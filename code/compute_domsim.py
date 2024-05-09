# compute domain similarity.
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import random

def extract_features(sentences):
    from transformers import AutoTokenizer, AutoModel
    import torch
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings

if __name__ == "__main__":
    datasets = ["AM_Movies", "AM_Electronics", "AM_CDs", "TripAdvisor", "Yelp"]
    random.seed(42)
    domain2sentences = {}
    for i, dataset in enumerate(datasets):
        df = pd.read_csv(f"./data/{dataset}/processed.csv")
        sentences = []
        for j in range(len(df)):
            prob = random.random()
            if prob > 0.5:
                sentences.append(df.iloc[j]["explanation"])
            # if len(sentences) == 5:
            if len(sentences) == 500:
                break
        domain2sentences[dataset] = sentences
        
    domain2embeddings = {}
    for i, dataset in enumerate(datasets):
        sentences = domain2sentences[dataset]
        sentence_embeddings = extract_features(sentences)
        domain2embeddings[dataset] = sentence_embeddings

    pairs = [
        ("AM_Movies", "AM_Electronics"),
        ("AM_Movies", "AM_CDs"),
        ("AM_Electronics", "AM_CDs"),
        ("AM_Movies", "TripAdvisor"),
        ("AM_Electronics", "TripAdvisor"),
        ("AM_CDs", "TripAdvisor"),
        ("AM_Movies", "Yelp"),
        ("AM_Electronics", "Yelp"),
        ("AM_CDs", "Yelp"),
        ("TripAdvisor", "Yelp")
    ]
    average_similarity = {}
    for pair in pairs:
        domain1, domain2 = pair
        # Compute cosine similarity
        similarity_matrix = cosine_similarity(domain2embeddings[domain1], domain2embeddings[domain2])
        # Exclude diagonal and compute average similarity
        np.fill_diagonal(similarity_matrix, np.nan)
        avg_sim = np.nanmean(similarity_matrix)
        average_similarity[f"{domain1} <-> {domain2}"] = avg_sim
    with open("./similarity.out", "w") as file:
        for pair, similarity in average_similarity.items():
            file.write(f"{pair}: {similarity}\n")
