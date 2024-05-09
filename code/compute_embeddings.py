import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm.auto import tqdm

def extract_embeddings_in_batches(encoded_inputs, model, batch_size=8):
    # Move model to device
    model.to(device)
    model.eval()

    # Prepare for batch processing
    total_batches = (encoded_inputs.input_ids.size(0) + batch_size - 1) // batch_size
    total_embeddings = []
    for i in tqdm(range(0, encoded_inputs.input_ids.size(0), batch_size), total=total_batches, desc="Processing batches"):
        # Process inputs in batches
        batch_input_ids = encoded_inputs.input_ids[i:i+batch_size].to(device)
        batch_attention_mask = encoded_inputs.attention_mask[i:i+batch_size].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
        
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        total_embeddings.append(batch_embeddings)

    # Concatenate all batch embeddings
    return np.concatenate(total_embeddings, axis=0)


if __name__ == "__main__":
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    datasets=["AM_CDs", "AM_Electronics", "AM_Movies", "TripAdvisor","Yelp"]
    # datasets=["Yelp"]
    for dataset in datasets:
        df = pd.read_csv(f"../../data/{dataset}/train.csv")
        nusers = df['user_idx'].max() + 1
        nitems = df['item_idx'].max() + 1

        # user embeddings
        grouped_reviews = df.groupby('user_idx')['review'].apply(lambda reviews: ' '.join(reviews))
        encoded_input = tokenizer(list(grouped_reviews), padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        embeddings = extract_embeddings_in_batches(encoded_input, model, batch_size=256)
        user_embeddings = np.random.rand(nusers, embeddings.shape[1])
        user_embeddings[grouped_reviews.index] = embeddings
        np.save(dataset+'_user_embeddings.npy', user_embeddings)
        
        # item embeddings
        grouped_reviews = df.groupby('item_idx')['review'].apply(lambda reviews: ' '.join(reviews))
        encoded_input = tokenizer(list(grouped_reviews), padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        embeddings = extract_embeddings_in_batches(encoded_input, model, batch_size=256)
        item_embeddings = np.random.rand(nitems, embeddings.shape[1])
        item_embeddings[grouped_reviews.index] = embeddings
        np.save(dataset+'_item_embeddings.npy', item_embeddings)
        
        
