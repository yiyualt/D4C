# import os
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


logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
    
    
class Processor():
    def __init__(self, train):
        self.max_length = 25
        self.train = train
    def __call__(self, sample):
        if self.train: 
            user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
            item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
            raitng = torch.tensor(sample["rating"], dtype=torch.float)
            explanation = sample["explanation"]
            counterfactual = sample["counterfactual"]
            try:
                explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            except:
                explanation_idx = tokenizer("no explanation", padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            try:
                counterfactual_idx = tokenizer(counterfactual, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            except:
                counterfactual_idx = tokenizer("no counterfactual", padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)
            counterfactual_idx = torch.tensor(counterfactual_idx, dtype=torch.long)
            return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx, "counterfactual_idx": counterfactual_idx}
        else:
            user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
            item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
            raitng = torch.tensor(sample["rating"], dtype=torch.float)
            explanation = sample["explanation"]
            try:
                explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            except:
                explanation_idx = tokenizer("no explanation", padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
            explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)
            return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx}
            
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

    def forward(self, hidden): 
        mlp_vector = self.sigmoid(self.linear1(hidden))
        rating = self.linear2(mlp_vector).view(-1)  
        return rating
    
class D4C(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, user_profiles, item_profiles, edited_user_features, edited_item_features):
        super().__init__()
        ntypes = 2  #determine if counterfactual or original samples
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.type_embeddings = nn.Embedding(ntypes, emsize)
        self.user_profiles = nn.Parameter(user_profiles)  
        self.item_profiles = nn.Parameter(item_profiles)
        self.edited_user_features = edited_user_features
        self.edited_item_features = edited_item_features
        self.recommender = PETER_MLP(emsize)
        self.hidden2token = nn.Linear(emsize, ntoken)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)  
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

    def gather(self, batch, device, train):
        if train: 
            user_idx, item_idx, rating, tgt_output, count_output = batch
            count_output = count_output.to(device)
            count_input = T5_shift_right(count_output)
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device).float()
            tgt_output = tgt_output.to(device)
            tgt_input = T5_shift_right(tgt_output)
            return user_idx, item_idx, rating, tgt_input, tgt_output, count_input, count_output
        else:
            user_idx, item_idx, rating, tgt_output = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            rating = rating.to(device).float()
            tgt_output = tgt_output.to(device)
            tgt_input = T5_shift_right(tgt_output)
            return user_idx, item_idx, rating, tgt_input, tgt_output

    def forward(self, user, item, tgt_input):
        device = user.device
        types = torch.zeros_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1) 
        user_embed = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_embed = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        word_feature = self.word_embeddings(tgt_input)   # in shape (N,seqlen, emsize)
        user_feature = torch.mean(torch.stack((user_embed, user_profile, user_embed*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_embed, item_profile, item_embed*item_profile)), dim=0)
        src = torch.cat([type_feature, user_feature, item_feature, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = generate_count_mask(tgt_input.shape[1], device)
        hidden = self.transformer_encoder(src=src, mask=attn_mask)
        rating = self.recommender(hidden[:,1])
        context_dist = self.hidden2token(hidden[:,2]).unsqueeze(1).repeat(1, tgt_input.shape[1], 1) 
        word_dist = self.hidden2token(hidden[:,3:])
        return rating, context_dist, word_dist

    def take_counterfactuals(self, user, item, count_input):
        device = user.device
        types = torch.ones_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1)
        user_feature = self.edited_user_features[user].unsqueeze(dim=1)
        item_feature = self.edited_item_features[item].unsqueeze(dim=1)
        word_feature = self.word_embeddings(count_input)
        src = torch.cat([type_feature, user_feature, item_feature, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        attn_mask = generate_count_mask(count_input.shape[1], device)
        hidden = self.transformer_encoder(src=src, mask=attn_mask)
        context_dist = self.hidden2token(hidden[:,2]).unsqueeze(1).repeat(1, count_input.shape[1], 1)
        word_dist = self.hidden2token(hidden[:,3:])
        return context_dist, word_dist

    def recommend(self, user, item):
        types = torch.zeros_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1) 
        user_embed = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_embed = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        user_feature = torch.mean(torch.stack((user_embed, user_profile, user_embed*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_embed, item_profile, item_embed*item_profile)), dim=0)
        src = torch.cat([type_feature, user_feature, item_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        hidden = self.transformer_encoder(src)
        rating = self.recommender(hidden[:,1])  
        return rating

    def generate(self, user, item):
        max_len = 25
        bos_idx = 0
        device = user.device
        batch_size = user.shape[0]
        types = torch.zeros_like(user)
        type_feature = self.type_embeddings(types).unsqueeze(dim=1) 
        user_embed = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_embed = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)      
        user_feature = torch.mean(torch.stack((user_embed, user_profile, user_embed*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_embed, item_profile, item_embed*item_profile)), dim=0)
        
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)   # in shape (N,1)
        for i in range(max_len):
            word_feature = self.word_embeddings(decoder_input_ids) 
            src = torch.cat([type_feature, user_feature, item_feature, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)  # in shape: (N, 2+1, emsize)
            attn_mask = generate_count_mask(decoder_input_ids.shape[1], device)
            hidden = self.transformer_encoder(src=src, mask=attn_mask)     # in shape (N, 3, emsize)
            dist = self.hidden2token(hidden).softmax(dim=-1)
            output_id = dist[:,-1,:].topk(1).indices                       # in shape (N, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
        return decoder_input_ids[:,1:]  # removing <BOS>


def trainModel(model, train_dataloader, valid_dataloader, config):
    epochs = config.get("epochs")
    learning_rate = config.get("learning_rate")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    coef = config.get("coef")
    weight = config.get("weight")
    device = config.get("device")
    ntoken = config.get("ntoken")
    log_file = config.get("log_file")
    save_file = config.get("save_file")
    prev_valid_loss = float("inf")
    enduration = 0
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            user_idx, item_idx, rating, tgt_input, tgt_output, count_input, count_output = model.gather(batch, device, True)
            pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input)
            count_context_dist, cont_word_dist = model.take_counterfactuals(user_idx, item_idx, count_input)
            loss_r = model.rating_loss_fn(pred_rating, rating)
            loss_e = model.exp_loss_fn(word_dist.view(-1, ntoken), tgt_output.reshape(-1))
            loss_c = model.exp_loss_fn(context_dist.view(-1, ntoken), tgt_output.reshape(-1))
            loss_c_c = model.exp_loss_fn(count_context_dist.view(-1, ntoken), count_output.reshape(-1))
            loss_c_w = model.exp_loss_fn(cont_word_dist.view(-1, ntoken), count_output.reshape(-1))
            loss_con = weight * (loss_c_c + loss_c_w)
            loss_reg = coef * (loss_r + loss_c)
            loss = loss_e + loss_con + loss_reg
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()
        avg_loss /= len(train_dataloader)
        with open(log_file, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
            f.write(f"Epoch {epoch+1}: [{current_time}] [lr: {learning_rate}] Loss = {avg_loss:.4f}\n")
        current_valid_loss = validModel(model, valid_dataloader, device)
        if current_valid_loss > prev_valid_loss:
            learning_rate = learning_rate * 0.5
            enduration +=1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            torch.save(model.state_dict(), save_file)
        prev_valid_loss = current_valid_loss
        if enduration >= 5:
            break

def validModel(model, valid_dataloader, device):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output = model.gather(batch, device, False)
            pred_rating, context_dist, word_dist = model(user_idx, item_idx, tgt_input)
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
            user_idx, item_idx, rating, tgt_input, tgt_output = model.gather(batch, device, False)
            pred_ratings = model.recommend(user_idx, item_idx)
            pred_exps = model.generate(user_idx, item_idx)
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
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_file", type=str, default= "model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--coef", type=float, default=0.5)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source", type=str, default="AM_Electronics")
    parser.add_argument("--target", type=str, default="AM_CDs")
    parser.add_argument("--weight", type=float, default=0.01)
    args = parser.parse_args()

    config = {"source": args.source,
            "target": args.target, 
            "device": args.device, 
            "log_file": args.log_file,
            "save_file": args.save_file,
            "learning_rate": args.learning_rate,
            "weight": args.weight,
            "coef": args.coef,
            "epochs": args.epochs,
            "seed": args.seed,
            "batch_size": 128, 
            "ntoken": 32128,
            "emsize": 768,
            "nhead": 2, 
            "nhid": 2048,
            "nlayers": 2, 
            "dropout": 0.2}

    pairs = [("AM_Electronics", "AM_CDs"),
            ("AM_Movies", "AM_CDs"),
            ("AM_CDs", "AM_Electronics"), 
            ("AM_Movies", "AM_Electronics"),
            ("AM_CDs", "AM_Movies"),
            ("AM_Electronics", "AM_Movies"), 
            ("Yelp", "TripAdvisor"),
            ("TripAdvisor", "Yelp")]
    index = pairs.index((config.get("source"), config.get("target"))) + 1
    train_path = f"../../data/counterfactual/{str(index)}"
    edited_user_features = torch.tensor(np.load(train_path+"/user_features.npy"), dtype=torch.float, device=config.get("device"))
    edited_item_features = torch.tensor(np.load(train_path+"/item_features.npy"), dtype=torch.float, device=config.get("device"))
    train_dataset = load_dataset("csv", data_files={"train": train_path+"/train.csv"})
    train_processor = Processor(train=True)
    encoded_data = train_dataset["train"].map(train_processor, batched=True)
    encoded_data.set_format(type="torch")
    dataset = TensorDataset(encoded_data['user_idx'],
                                encoded_data['item_idx'],
                                encoded_data['rating'],
                                encoded_data['explanation_idx'], 
                                encoded_data['counterfactual_idx'])
    train_dataloader = DataLoader(dataset, batch_size=config.get("batch_size"), shuffle=True)


    path = f"../../data/{config.get('target')}/"
    eval_dataset = load_dataset("csv", data_files={"valid": path+"/valid.csv", "test": path+"/test.csv"})
    user_profiles = torch.tensor(np.load(path+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
    item_profiles = torch.tensor(np.load(path+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
    eval_processor = Processor(train=False)
    encoded_data = eval_dataset.map(eval_processor, batched=True)
    encoded_data.set_format(type="torch")
    valid_dataset = TensorDataset(encoded_data["valid"]['user_idx'],
                                encoded_data["valid"]['item_idx'],
                                encoded_data["valid"]['rating'],
                                encoded_data["valid"]['explanation_idx'])
                                
    test_dataset = TensorDataset(encoded_data["test"]['user_idx'],
                                encoded_data["test"]['item_idx'],
                                encoded_data["test"]['rating'],
                                encoded_data["test"]['explanation_idx'])
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.get("batch_size"), shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    nuser = max(train_dataset["train"]["user_idx"]) + 1
    nitem = max(train_dataset["train"]["item_idx"]) + 1
    model = D4C(nuser, nitem, 
                config["ntoken"],
                config["emsize"],
                config["nhead"], 
                config["nhid"], 
                config["nlayers"],
                config["dropout"], 
                user_profiles, 
                item_profiles, 
                edited_user_features, 
                edited_item_features).to(config["device"])
    
    with open(config.get("log_file"), "a") as f:
        f.write(f"\nCurrent time: {datetime.now()}\n")
        f.write(f"\nConfig: {config}\n")
        
    trainModel(model, train_dataloader, valid_dataloader, config)
    model.load_state_dict(torch.load(config.get("save_file")))
    final = evalModel(model, test_dataloader, config.get("device"))
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