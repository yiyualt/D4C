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
import torch.nn.init as init
import torch.nn.functional as F
import itertools


logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=True)
    
class Processor():
    def __init__(self):
        self.max_length = 25
    def __call__(self, sample):
        user_idx = torch.tensor(sample["user_idx"], dtype=torch.long)
        item_idx = torch.tensor(sample["item_idx"], dtype=torch.long)
        raitng = torch.tensor(sample["rating"], dtype=torch.float)
        explanation = sample["explanation"]
        explanation_idx = tokenizer(explanation, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]
        explanation_idx = torch.tensor(explanation_idx, dtype=torch.long)
        return {"user_idx": user_idx, "item_idx": item_idx, "rating": raitng, "explanation_idx": explanation_idx}


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super(Discriminator, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        
        self.fc4 = nn.Linear(hidden_size, 1)  # Output 1 if from source domain, 0 if from target domain
        self.init_weights()

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x.view(-1)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.02)  # Mean and standard deviation can be adjusted as needed
                init.constant_(m.bias, 0)

                
class Editor(nn.Module):
    def __init__(self, hidden_size):
        super(Editor, self).__init__()
        self.hidden_size = hidden_size
        self.user_edit = nn.Sequential (nn.Linear(hidden_size, hidden_size),
                                        nn.LayerNorm(hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.LayerNorm(hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size)
        )
        self.item_edit = nn.Sequential (nn.Linear(hidden_size, hidden_size),
                                        nn.LayerNorm(hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.LayerNorm(hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, hidden_size)
        )
        self.init_weights()

    def forward(self, user_feature, item_feature):
        user_edited = torch.tanh(self.user_edit(user_feature))
        item_edited = torch.tanh(self.item_edit(item_feature))
        return user_edited, item_edited
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, mean=0, std=0.02)  # Mean and standard deviation can be adjusted as needed
                init.constant_(m.bias, 0)

class BCEWithLogitsLoss(nn.Module):
    def __init__(self, label_smoothing=0.0, reduction='mean'):
        super(BCEWithLogitsLoss, self).__init__()
        assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        if self.label_smoothing > 0:
            positive_smoothed_labels = 1.0 - self.label_smoothing
            negative_smoothed_labels = self.label_smoothing
            target = target * positive_smoothed_labels + \
                (1 - target) * negative_smoothed_labels

        loss = self.bce_with_logits(input, target)
        return loss


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
        
    
class Generator(nn.Module):
    def __init__(self, nuser, nitem, ntoken, emsize, nhead, nhid, nlayers, dropout, user_profiles, item_profiles):
        super().__init__()
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.word_embeddings = nn.Embedding(ntoken, emsize)
        self.user_profiles = nn.Parameter(user_profiles)  # user_profiles
        self.item_profiles = nn.Parameter(item_profiles)
        self.hidden2token = nn.Linear(emsize, ntoken)
        self.recommender = PETER_MLP(emsize)
        encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True)  # nhid: dim_feedforward, one 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)  # loop over the one above
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        self.emsize = emsize
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
    
    def forward(self, user, item, tgt_input):
        device = user.device
        user_feature = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_feature = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)
        word_feature = self.word_embeddings(tgt_input)   # in shape (N,seqlen, emsize)
        user_feature = torch.mean(torch.stack((user_feature, user_profile, user_feature*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_feature, item_profile, item_feature*item_profile)), dim=0)
        src = torch.cat([user_feature, item_feature, word_feature], dim=1)
        src = src * math.sqrt(self.emsize)
        src = self.pos_encoder(src)
        
        # peter mask and pad mask
        attn_mask = generate_peter_mask(tgt_input.shape[1], device)
        hidden = self.transformer_encoder(src=src, mask=attn_mask)
        context_dist = self.hidden2token(hidden[:,1]).unsqueeze(1).repeat(1, tgt_input.shape[1], 1) 
        word_dist = self.hidden2token(hidden[:,2:])
        return context_dist, word_dist  # (N), (N,seqlen,emsize), (N,seqlen,emsize) respectively
    
    def gather(self, batch, device):
        user_idx, item_idx, rating, tgt_output = batch
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        rating = rating.to(device).float()
        tgt_output = tgt_output.to(device)
        tgt_input = T5_shift_right(tgt_output)
        return user_idx, item_idx, rating, tgt_input, tgt_output
    
    def generate(self, user, item):
        max_len = 25
        bos_idx = 0
        device = user.device
        batch_size = user.shape[0]
        user_feature = self.user_embeddings(user).unsqueeze(dim=1)  # in shape (N,1, emsize)
        item_feature = self.item_embeddings(item).unsqueeze(dim=1)  # in shape (N,1, emsize)
        user_profile = self.user_profiles[user].unsqueeze(dim=1)
        item_profile = self.item_profiles[item].unsqueeze(dim=1)      
        user_feature = torch.mean(torch.stack((user_feature, user_profile, user_feature*user_profile)), dim=0)
        item_feature = torch.mean(torch.stack((item_feature, item_profile, item_feature*item_profile)), dim=0)
        
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)   # in shape (N,1)
        for i in range(max_len):
            word_feature = self.word_embeddings(decoder_input_ids) 
            src = torch.cat([user_feature, item_feature, word_feature], dim=1)
            src = src * math.sqrt(self.emsize)
            src = self.pos_encoder(src)  # in shape: (N, 2+1, emsize)
            attn_mask = generate_peter_mask(decoder_input_ids.shape[1], device)
            hidden = self.transformer_encoder(src=src, mask=attn_mask)     # in shape (N, 3, emsize)
            dist = self.hidden2token(hidden).softmax(dim=-1)
            output_id = dist[:,-1,:].topk(1).indices                       # in shape (N, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
        return decoder_input_ids[:,1:]  # removing <BOS>
        

class EmbeddingModel(nn.Module):
    def __init__(self, user_embeddings, item_embeddings, user_profiles, item_profiles):
        super(EmbeddingModel, self).__init__()
        self.user_embeddings = user_embeddings
        self.user_profiles = user_profiles
        self.item_embeddings = item_embeddings
        self.item_profiles = item_profiles

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embeddings(user_indices)  #(N, emsize)
        item_embeds = self.item_embeddings(item_indices)  #(N, emsize)
        user_profile = self.user_profiles[user_indices]   #(N, emsize)
        item_profile = self.item_profiles[item_indices]   #(N, emsize)
        user_features = torch.mean(torch.stack((user_embeds, user_profile, user_embeds*user_profile)), dim=0)
        item_features = torch.mean(torch.stack((item_embeds, item_profile, item_embeds*item_profile)), dim=0)
        return user_features, item_features


def choose_Gumbel(A,B):
    """
    A is auxiliary features, and B is replacing features. 
    """
    device = A.device
    A_normalized = F.normalize(A, p=2, dim=1)
    B_normalized = F.normalize(B, p=2, dim=1)
    cosine_sim = torch.mm(A_normalized, B_normalized.transpose(0, 1))  # Resulting shape (N, M)

    tau = 0.5
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(cosine_sim)))
    gumbel_scores = (cosine_sim + gumbel_noise) / tau
    softmax_scores = F.softmax(gumbel_scores, dim=1)
    _, indices = torch.max(softmax_scores, dim=1)
    return B[indices]

        
def trainModel(embedding_model, Ga_model, disc_model, editor, train_dataloader, strain_dataloader, svalid_dataloader, config):
    lr = config.get("learning_rate")
    epochs = config.get("epochs")
    coef = config.get("coef")
    device = config.get("device")
    prob = config.get("prob")
    weight_decay = config.get("weight_decay")
    disc_fn = BCEWithLogitsLoss(label_smoothing=0.1)
    editor_optimizer = optim.Adam(editor.parameters(), lr=lr, weight_decay=weight_decay)
    disc_optimizer = optim.Adam(disc_model.parameters(), lr=lr, weight_decay=1e-5)
    Ga_optimizer = optim.Adam(Ga_model.parameters(), lr=lr, weight_decay=1e-5)
    enduration = 0
    prev_valid_loss = float("inf")

    for epoch in range(epochs):
        Ga_model.train()
        disc_model.train()
        editor.train()
        avg_loss = 0
        source_iter = iter(strain_dataloader)
        target_iter = iter(train_dataloader)
        prev_sbatch = None
        prev_tbatch = None
        total_batches = max(len(train_dataloader), len(strain_dataloader))
        for tbatch, sbatch in tqdm(itertools.zip_longest(target_iter, source_iter), total=total_batches):
            if tbatch is None:
                tbatch = prev_tbatch
            else:
                prev_tbatch = tbatch
            if sbatch is None:
                sbatch = prev_sbatch
            else:
                prev_sbatch = sbatch
            # train disc_model. Step A. 
            for p in disc_model.parameters():
                p.requires_grad = True
            for p in Ga_model.parameters():
                p.requires_grad = False
            for p in editor.parameters():
                p.requires_grad = False
            user_idx, item_idx = tbatch
            user_idx = user_idx.to(config.get("device"))
            item_idx = item_idx.to(config.get("device"))
            user_features, item_features = embedding_model(user_idx, item_idx)
            user_edited, item_edited = editor(user_features, item_features)
            edited_features = torch.cat((user_edited, item_edited), dim=0) # (2N, k)
            target_labels = torch.zeros(len(edited_features)).to(device)    
            loss1 = disc_fn(disc_model(edited_features), target_labels)

            suser_idx, sitem_idx, srating, stgt_input, stgt_output = Ga_model.gather(sbatch, config["device"])
            suser_embeds = Ga_model.user_embeddings(suser_idx)
            sitem_embeds = Ga_model.item_embeddings(sitem_idx)
            suser_profile = Ga_model.user_profiles[suser_idx]
            sitem_profile = Ga_model.item_profiles[sitem_idx]
            suser_feature = torch.mean(torch.stack((suser_embeds, suser_profile, suser_embeds*suser_profile)), dim=0)
            sitem_feature = torch.mean(torch.stack((sitem_embeds, sitem_profile, sitem_embeds*sitem_profile)), dim=0)
            source_features = torch.cat((suser_feature, sitem_feature), dim=0)
            source_labels = torch.ones(len(source_features)).to(device)
            loss2 = disc_fn(disc_model(source_features), source_labels)
            loss = loss1 + loss2
            loss.backward()
            nn.utils.clip_grad_norm_(disc_model.parameters(), 1)
            disc_optimizer.step()
            disc_optimizer.zero_grad()

            # train Editor and Ga_model. Step B. 
            for p in editor.parameters():
                p.requires_grad = True
            for p in Ga_model.parameters():
                p.requires_grad = True
            for p in disc_model.parameters():
                p.requires_grad = False
            user_edited, item_edited = editor(user_features, item_features) # editor has gradient now
            edited_features = torch.cat((user_edited, item_edited), dim=0)  # (2N, k)
            target_labels = torch.ones(len(edited_features)).to(device)
            tdomain_pred = disc_model(edited_features)
            loss_d = disc_fn(tdomain_pred, target_labels)

            suser_embeds = Ga_model.user_embeddings(suser_idx)  # in shape (N, k)
            sitem_embeds = Ga_model.item_embeddings(sitem_idx) # in shape (N, k)
            suser_profile = Ga_model.user_profiles[suser_idx]
            sitem_profile = Ga_model.item_profiles[sitem_idx]
            suser_feature = torch.mean(torch.stack((suser_embeds, suser_profile, suser_embeds*suser_profile)), dim=0)
            sitem_feature = torch.mean(torch.stack((sitem_embeds, sitem_profile, sitem_embeds*sitem_profile)), dim=0)
            if torch.rand(1).item() > prob:  # use edited feature instead.
                final_suser_feature = choose_Gumbel(suser_feature, user_edited)
            else: 
                final_suser_feature = suser_feature
            if torch.rand(1).item() > prob: 
                final_sitem_feature = choose_Gumbel(sitem_feature, item_edited)
            else:
                final_sitem_feature = sitem_feature         
            word_feature = Ga_model.word_embeddings(stgt_input)       
            src = torch.cat([final_suser_feature.unsqueeze(1), final_sitem_feature.unsqueeze(1), word_feature], dim=1)
            src = src * math.sqrt(Ga_model.emsize)
            src = Ga_model.pos_encoder(src)
            attn_mask = generate_peter_mask(stgt_input.shape[1], device)
            hidden = Ga_model.transformer_encoder(src=src, mask=attn_mask)
            context_dist = Ga_model.hidden2token(hidden[:,1]).unsqueeze(1).repeat(1, stgt_input.shape[1], 1) 
            word_dist = Ga_model.hidden2token(hidden[:,2:])
            loss_e = Ga_model.exp_loss_fn(word_dist.view(-1, 32128), stgt_output.reshape(-1))
            loss_c = Ga_model.exp_loss_fn(context_dist.view(-1, 32128), stgt_output.reshape(-1))
            loss = coef*loss_d + loss_e + loss_c
            loss.backward()
            nn.utils.clip_grad_norm_(editor.parameters(), 1)
            nn.utils.clip_grad_norm_(Ga_model.parameters(), 1)
            editor_optimizer.step()
            editor_optimizer.zero_grad()
            Ga_optimizer.step()
            Ga_optimizer.zero_grad()
            avg_loss += loss.item()
        avg_loss /= total_batches
        with open(config["log_file"], "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
            f.write(f"Epoch {epoch+1}: [{current_time}] [lr: {lr}] Loss = {avg_loss:.4f}\n")

        # checking learning rate
        current_valid_loss = validModel(Ga_model, svalid_dataloader, device)
        if current_valid_loss > prev_valid_loss:
            lr /= 2.0
            enduration += 1
            for param_group in disc_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in editor_optimizer.param_groups:
                param_group['lr'] = lr
            for param_group in Ga_optimizer.param_groups:
                param_group['lr'] = lr
        else:
            torch.save(Ga_model.state_dict(), "G_"+config.get("save_file"))
            torch.save(editor.state_dict(), "editor_"+config.get("save_file"))

        prev_valid_loss = current_valid_loss
        if enduration  >= 5:
            break

def validModel(model, valid_dataloader, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            user_idx, item_idx, rating, tgt_input, tgt_output = model.gather(batch, device)
            context_dist, word_dist = model(user_idx, item_idx, tgt_input)  
            loss_e = model.exp_loss_fn(word_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss_c = model.exp_loss_fn(context_dist.view(-1, 32128), tgt_output.reshape(-1))
            loss = loss_c + loss_e
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


def evalModel(embedding_model, Ga_model, editor, test_dataloader, device):
    def compute_entropy(generated_dist):
        log_probabilities = torch.log(generated_dist + 1e-9)
        entropies = -torch.sum(generated_dist * log_probabilities, dim=-1)
        average_entropy = torch.mean(entropies)
        return average_entropy
    bos_idx = 0
    max_len = 25
    embedding_model = embedding_model.to(device)
    Ga_model = Ga_model.to(device)
    editor = editor.to(device)
    embedding_model.eval()
    Ga_model.eval()
    editor.eval()
    total_entropy = 0
    with torch.no_grad():
        for batch in test_dataloader:
            user_idx, item_idx = batch
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            user_features, item_features = embedding_model(user_idx,item_idx)
            user_edited, item_edited = editor(user_features, item_features)
            user_edited = user_edited.unsqueeze(1)
            item_edited = item_edited.unsqueeze(1)

            # generate counterfactua
            batch_size = user_edited.shape[0]
            decoder_input_ids = torch.zeros((batch_size, 1)).fill_(bos_idx).long().to(device)   # in shape (N,1)
            for i in range(max_len):
                word_feature = Ga_model.word_embeddings(decoder_input_ids) 
                src = torch.cat([user_edited, item_edited, word_feature], dim=1)
                src = src * math.sqrt(Ga_model.emsize)
                src = Ga_model.pos_encoder(src)  # in shape: (N, 2+1, emsize)
                attn_mask = generate_peter_mask(decoder_input_ids.shape[1], device)
                hidden = Ga_model.transformer_encoder(src=src, mask=attn_mask)     # in shape (N, 3, emsize)
                dist = Ga_model.hidden2token(hidden).softmax(dim=-1)
                output_id = dist[:,-1,:].topk(1).indices                       # in shape (N, 1)
                decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
                total_entropy += compute_entropy(dist).item()
        return total_entropy / len(test_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="log.out")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source", type=str, default="AM_Electronics")
    parser.add_argument("--target", type=str, default="AM_CDs")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--save_file", type=str, default= "model.pth")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coef", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--prob", type=float, default=0.9)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    # load target
    config = {
        "dataset": args.target,
        "device": "cuda",
        "log_file": args.log_file,
        "save_file":args.save_file,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": 128,
        "emsize": 768,
        "nlayers": 2,
        "nhid": 2048,
        "ntoken": 32128,
        "dropout": 0.2,
        "nhead": 2, 
        "coef": args.coef, 
        "weight_decay": args.weight_decay,
        "prob": args.prob,
        "seed": args.seed
        }
    # for target domain, we need prepare target user and item embeddings and data for test.  
    path = "../../data/"+args.target
    train_df = pd.read_csv(path+"/train.csv")
    nuser = train_df['user_idx'].max() + 1
    nitem = train_df['item_idx'].max() + 1
    user_profiles = torch.tensor(np.load(path+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
    item_profiles = torch.tensor(np.load(path+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
    Gt_model = Generator(nuser, nitem, config.get("ntoken"), config.get("emsize"), config.get("nhead"), config.get("nhid"), config.get("nlayers"), config.get("dropout"), user_profiles, item_profiles)
    Gt_model.load_state_dict(torch.load("./saved_model/"+args.target+".pth"))
    user_embeddings = Gt_model.user_embeddings
    item_embeddings = Gt_model.item_embeddings
    embedding_model = EmbeddingModel(user_embeddings, item_embeddings, user_profiles, item_profiles).to(config.get("device"))
    for param in embedding_model.parameters():
        param.requires_grad = False

    user_tensor = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
    item_tensor = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
    train_dataset = TensorDataset(user_tensor, item_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=config.get("batch_size"), shuffle=True)
    
    # load source data and model.
    spath = "../../data/"+args.source
    strain_df = pd.read_csv(spath+"/train.csv")
    snuser = strain_df['user_idx'].max() + 1
    snitem = strain_df['item_idx'].max() + 1
    sdatasets = load_dataset("csv", data_files={"train": spath+"/train.csv", "valid": spath+"/valid.csv", "test": spath+"/test.csv"})
    sprocessor = Processor()
    sencoded_data = sdatasets.map(lambda sample: sprocessor(sample))
    sencoded_data.set_format("torch")
    strain_dataset = TensorDataset(sencoded_data['train']['user_idx'],
                              sencoded_data['train']['item_idx'],
                              sencoded_data['train']['rating'],
                              sencoded_data['train']['explanation_idx'])

    svalid_dataset = TensorDataset(sencoded_data['valid']['user_idx'],
                                  sencoded_data['valid']['item_idx'],
                                  sencoded_data['valid']['rating'],
                                  sencoded_data['valid']['explanation_idx'])

    strain_dataloader = DataLoader(strain_dataset, batch_size=config.get("batch_size"), shuffle=True)
    svalid_dataloader = DataLoader(svalid_dataset, batch_size=config.get("batch_size"), shuffle=True)
    suser_profiles = torch.tensor(np.load(spath+"/user_profiles.npy"), dtype=torch.float, device=config.get("device"))
    sitem_profiles = torch.tensor(np.load(spath+"/item_profiles.npy"), dtype=torch.float, device=config.get("device"))
    Ga_model = Generator(snuser, snitem, config.get("ntoken"), config.get("emsize"), config.get("nhead"), config.get("nhid"), config.get("nlayers"), config.get("dropout"), suser_profiles, sitem_profiles).to(config.get("device"))
    with open(config.get("log_file"), "a") as f:
        f.write(f"\nCurrent time: {datetime.now()}")
        f.write(f"\nConfig: {config}\n")
    
    disc_model = Discriminator(config.get("emsize")).to(config.get("device"))
    editor = Editor(config.get("emsize")).to(config.get("device"))
    # train model
    trainModel(embedding_model, Ga_model, disc_model, editor, train_dataloader, strain_dataloader, svalid_dataloader, config)
    Ga_model.load_state_dict(torch.load("G_"+config.get("save_file")))
    editor.load_state_dict(torch.load("editor_"+config.get("save_file")))

    # use valid or test.
    final = evalModel(embedding_model, Ga_model, editor, train_dataloader, config.get("device"))
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    with open(config.get("log_file"), "a") as f:
        f.write("------------------------------------------FINAL RESULTS------------------------------------------\n")
        f.write(f"[{current_time}] \n")
        f.write(f"final entropy: {round(final,2)}\n")
        logging.info("DONE.")