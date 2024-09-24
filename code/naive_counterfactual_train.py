import os
import sys
sys.path.append("../..")
from base_utils import * 
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import logging
from transformers import T5Tokenizer
from torch import nn, optim
import argparse
from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import numpy as np

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)

tasks = [
    ("AM_Electronics", "AM_CDs"),
    ("AM_Movies", "AM_CDs"),
    ("AM_CDs", "AM_Electronics"),
    ("AM_Movies", "AM_Electronics"),
    ("AM_CDs", "AM_Movies"),
    ("AM_Electronics", "AM_Movies"),
    ("Yelp", "TripAdvisor"),
    ("TripAdvisor", "Yelp")
]

class Processor():
    def __init__(self, auxiliary, target):
        self.max_length = 25
        self.auxiliary = auxiliary
        self.target = target

    def __call__(self, sample):
        user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
        item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
        raitng = torch.tensor(sample["rating"], dtype=torch.float)
        explanation = sample["explanation"]
        explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
        explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)

        if sample["domain"] == "auxiliary":
            domain_val = 0  # Auxiliary domain
        elif sample["domain"] == "target":
            domain_val = 1  # Target domain
        else:
            raise ValueError("Unknown domain!")

        domain_idx = torch.tensor(domain_val, dtype=torch.long)
        return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx, "domain_idx": domain_idx}

class PETER_MLP(nn.Module):
    def __init__(self, emsize=512):
        super().__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.weight.data.uniform_(-initrange, initrange)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        mlp_vector = self.sigmoid(self.linear1(hidden))  # (batch_size, emsize)
        rating = self.linear2(mlp_vector).view(-1)  # (batch_size,)
        return rating


class Model(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, user_profiles, item_profiles, domain_profiles):
        super().__init__()
        self.domain_profiles = nn.Parameter(domain_profiles)
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.user_profiles = nn.Parameter(user_profiles)  # user_profiles
        self.item_profiles = nn.Parameter(item_profiles)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.recommender = PETER_MLP(emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True)  # nhid: dim_feedforward, one 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)
    
    def forward(self, user, item, tgt_input, domain_idx):
        device = user.device
        domain_embedding = self.domain_profiles[domain_idx].unsqueeze(dim=1)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        word_feature = self.word_embeddings(tgt_input)   # in shape (N,seqlen, emsize)
        src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        
        # peter mask and pad mask
        attn_mask = generate_domain_mask(tgt_input.shape[1], device)
        hidden = self.transformer_encoder(src=src, mask=attn_mask)
        rating = self.recommender(hidden[:,3])
        context_dist = self.hidden2token(hidden[:,4]).unsqueeze(1).repeat(1, tgt_input.shape[1], 1) 
        word_dist = self.hidden2token(hidden[:,5:])
        return rating, context_dist, word_dist  # (N), (N,seqlen,emsize), (N,seqlen,emsize) respectively
    
    def gather(self, batch, device):
        user_idx, item_idx, rating, tgt_output, domain_idx = batch
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        domain_idx = domain_idx.to(device)
        rating = rating.to(device).float()
        tgt_output = tgt_output.to(device)
        tgt_input = T5_shift_right(tgt_output)
        return user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx

    def recommend(self, user, item, domain):
        domain_embedding = self.domain_profiles[domain].unsqueeze(dim=1)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden = self.transformer_encoder(src)
        rating = self.recommender(hidden[:,3])  
        return rating
    
    def generate(self, user, item, domain):
        max_len = 25
        bos_idx = 0
        device = user.device
        batch_size = user.shape[0]
        domain_embedding = self.domain_profiles[domain].unsqueeze(dim=1)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)      
        user_embeddings = self.user_embeddings(user).unsqueeze(dim=1)
        item_embeddings = self.item_embeddings(item).unsqueeze(dim=1)
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)   # in shape (N,1)
        for i in range(max_len):
            word_feature = self.word_embeddings(decoder_input_ids) 
            src = torch.cat([domain_embedding, user_profile, item_profile, user_embeddings, item_embeddings, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)  # in shape: (N, 2+1, emsize)
            attn_mask = generate_domain_mask(decoder_input_ids.shape[1], device)
            hidden = self.transformer_encoder(src=src, mask=attn_mask)     # in shape (N, 3, emsize)
            dist = self.hidden2token(hidden).softmax(dim=-1)
            output_id = dist[:,-1,:].topk(1).indices                       # in shape (N, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
        return decoder_input_ids[:,1:]  # removing <BOS>


def trainModel(model, train_dataloader, valid_dataloader, config):
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    coef = config["coef"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    enduration = 0
    prev_valid_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = model.gather(batch, config["device"])
            pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = model.rating_loss_fn(pred_rating, rating)
            loss_e = model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss_c = model.exp_loss_fn(context_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = coef*loss_r + coef*loss_c + loss_e
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()
        avg_loss /= len(train_dataloader)
        with open(config["log_file"], "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
            f.write(f"Epoch {epoch+1}: [{current_time}] [lr: {learning_rate}] Loss = {avg_loss:.4f}\n")

        # checking learning rate
        current_valid_loss = validModel(model, valid_dataloader, config.get("device"))
        if current_valid_loss > prev_valid_loss:
            learning_rate /= 2.0
            enduration += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            torch.save(model.state_dict(), config.get("save_file"))

        prev_valid_loss = current_valid_loss
        if enduration  >= 5:
            break


def validModel(model, valid_dataloader, device):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = model.gather(batch, device)
            pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input, domain_idx)
            loss_r = model.rating_loss_fn(pred_rating, rating)
            loss_e = model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = loss_r + loss_e
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


def evalModel(model, test_dataloader, device):
    model = model.to(device)
    model.eval()
    prediction_ratings = []
    ground_truth_ratings = []
    prediction_exps = []
    reference_exps = []
    with torch.no_grad():
        for batch in test_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output, domain_idx = model.gather(batch, device)
            pred_ratings = model.recommend(user_idx, item_idx, domain_idx)
            pred_exps = model.generate(user_idx, item_idx,  domain_idx)
            prediction_ratings.extend(pred_ratings.tolist())
            ground_truth_ratings.extend(rating.tolist())
            prediction_exps.extend(tokenizer.batch_decode(pred_exps, skip_special_tokens=True))
            reference_exps.extend(tokenizer.batch_decode(tgt_output, skip_special_tokens=True))

    prediction_ratings  = np.array(prediction_ratings)
    ground_truth_ratings = np.array(ground_truth_ratings)
    rating_diffs = prediction_ratings - ground_truth_ratings
    mae = round(np.mean(np.abs(rating_diffs)), 4)
    rmse = round(np.sqrt(np.mean(np.square(rating_diffs))),4)
    text_results = evaluate_text(prediction_exps, reference_exps)
    return {"recommendation": {"mae":mae, "rmse":rmse}, "explanation":text_results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="log.out")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--auxiliary", type=str, required=True) 
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_file", type=str, default="model.pth")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--coef", type=float, default=0.5)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task_idx = None
    for idx, (aux, tgt) in enumerate(tasks):
        if aux == args.auxiliary and tgt == args.target:
            task_idx = idx + 1
            break
    path = "../../Merged_data/"+str(task_idx)
    train_df = pd.read_csv(path+"/aug_train.csv")
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1

    config = {
        "task_idx": task_idx,
        "device": args.device,
        "log_file": args.log_file,
        "save_file":args.save_file,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": 128,
        "emsize": 768,
        "nlayers": args.nlayers,
        "nhid": 2048,
        "ntoken": 32128,
        "dropout": 0.2,
        "nuser": nuser,
        "nitem": nitem, 
        "coef": args.coef, 
        "nhead": 2
        }

    datasets = load_dataset("csv", data_files={"train": path+"/aug_train.csv", "valid": path+"/aug_valid.csv"})
    processor = Processor(args.auxiliary, args.target)
    encoded_data = datasets.map(lambda sample: processor(sample))
    encoded_data.set_format("torch")
    train_dataset = TensorDataset(encoded_data['train']['user_idx'],
                              encoded_data['train']['item_idx'],
                              encoded_data['train']['rating'],
                              encoded_data['train']['explanation_idx'],
                              encoded_data['train']['domain_idx']
                              )

    valid_dataset = TensorDataset(encoded_data['valid']['user_idx'],
                                  encoded_data['valid']['item_idx'],
                                  encoded_data['valid']['rating'],
                                  encoded_data['valid']['explanation_idx'], 
                                  encoded_data['valid']['domain_idx'])


    train_dataloader = DataLoader(train_dataset, batch_size=config.get("batch_size"), shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.get("batch_size"), shuffle=True)

    # load domain, user, item profiles from auxliary 
    suser_profiles = torch.tensor(np.load("../../data/"+args.auxiliary+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
    sitem_profiles = torch.tensor(np.load("../../data/"+args.auxiliary+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
    sdomain_profiles = torch.tensor(np.load("../../data/"+args.auxiliary+"/domain.npy"), dtype=torch.float, device=config.get("device"))

    # load target
    tuser_profiles = torch.tensor(np.load("../../data/"+args.target+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
    titem_profiles = torch.tensor(np.load("../../data/"+args.target+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
    tdomain_profiles = torch.tensor(np.load("../../data/"+args.target+"/domain.npy"), dtype=torch.float, device=config.get("device"))

    domain_profiles = torch.cat([sdomain_profiles.unsqueeze(0), tdomain_profiles.unsqueeze(0)], dim=0)
    user_profiles = torch.cat([tuser_profiles, suser_profiles], dim=0)
    item_profiles = torch.cat([titem_profiles, sitem_profiles], dim=0)

    model = Model(config.get("nuser"), config.get("nitem"), config.get("ntoken"), config.get("emsize"), config.get("nhead"), config.get("nhid"), config.get("nlayers"), config.get("dropout"), user_profiles, item_profiles, domain_profiles).to(config.get("device"))

    with open(config.get("log_file"), "a") as f:
        f.write(f"\nCurrent time: {datetime.now()}\n")
        f.write(f"\nConfig: {config}\n")

    trainModel(model, train_dataloader, valid_dataloader, config)
    model.load_state_dict(torch.load(config.get("save_file")))

    # use valid or test.
    final = evalModel(model, valid_dataloader, config.get("device"))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
    with open(config.get("log_file"), "a") as f:
        f.write("------------------------------------------FINAL RESULTS------------------------------------------\n")
        f.write(f"[{current_time}] \n")
        f.write(f"[Recommendation] MAE = {final['recommendation']['mae']} | RMSE = {final['recommendation']['rmse']} \n")
        f.write(f"[Explanation] ROUGE: {final['explanation']['rouge']['1']}, {final['explanation']['rouge']['2']}, {final['explanation']['rouge']['l']} \n")
        f.write(f"[Explanation] BLEU: {final['explanation']['bleu']['1']}, {final['explanation']['bleu']['2']}, {final['explanation']['bleu']['3']}, {final['explanation']['bleu']['4']} \n")
        f.write(f"[Explanation] DIST: {final['explanation']['dist']['1']}, {final['explanation']['dist']['2']},\n")
        f.write(f"[Explanation] METEOR: {final['explanation']['meteor']} \n")
        f.write(f"[Explanation] BERT: {final['explanation']['bert']} \n")
        logging.info("DONE.")