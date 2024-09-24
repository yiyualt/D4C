import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

def compute_embedding(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def split_into_chunks(text, max_chunk_length=512):
    words = text.split()
    for i in range(0, len(words), max_chunk_length):
        yield " ".join(words[i:i + max_chunk_length])

data_folder = './data'
dataset_folders = ["AM_Electronics", "AM_CDs", "AM_Movies", "TripAdvisor", "Yelp"]

for dataset in dataset_folders:
    file_path = os.path.join(data_folder, dataset, 'train.csv')
    data = pd.read_csv(file_path)

    aggregated_reviews = " ".join(data['review'].astype(str).tolist())

    chunks = list(split_into_chunks(aggregated_reviews))

    chunk_embeddings = []
    for chunk in tqdm(chunks, desc=f"Processing {dataset}"):
        embedding = compute_embedding([chunk], tokenizer, model)
        chunk_embeddings.append(embedding)

    domain_embedding = torch.stack(chunk_embeddings).mean(dim=0).squeeze().cpu().numpy()

    output_file = os.path.join(data_folder, dataset, 'domain.npy')
    np.save(output_file, domain_embedding)
    print(f"Domain embedding saved to {output_file}")
