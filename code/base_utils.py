from rouge import rouge
from nltk import word_tokenize
import evaluate
from bleu import compute_bleu
import numpy as np
import nltk
from torch import nn
import torch
import math

def T5_shift_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert decoder_start_token_id is not None, (
        "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
        " See T5 docs for more information"
    )
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id
    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


            
def evaluate_text(predictions, references):
    """
    Example:
        >>> predictions = ["good day", "need to work"]
        >>> references = ["nice day", "work from home"]
        >>> evlauate_text(predictions, references)
    """
    # compute bleu
    # compute rouge
    # compute distinct
    # compute meteor
    
    def distinct_score(sentences, n):
        sentences = [word_tokenize(sentence) for sentence in sentences]
        unique_ngrams = set()
        total_ngrams = 0

        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
            unique_ngrams.update(ngrams)
            total_ngrams += len(ngrams)

        distinct_score = len(unique_ngrams) / total_ngrams
        return distinct_score
    # dist score
    try:
        dist1 = round(distinct_score(predictions, 1) * 100, 2)
    except:
        dist1 = 0
    try:
        dist2 = round(distinct_score(predictions, 2) * 100, 2)
    except:
        dist2 = 0
    
    # bleu score
    predictions_tokens = [word_tokenize(prediction) for prediction in predictions]
    references_tokens = [word_tokenize(reference) for reference in references]
    formatted_ref = [[ref] for ref in references_tokens]
    try:
        bleu1, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=1, smooth=False)
        bleu1 = round(bleu1*100, 2)
    except:
        bleu1 = 0
    try:
        bleu2, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=2, smooth=False)
        bleu2 = round(bleu2*100, 2)
    except:
        bleu2 = 0
    try:
        bleu3, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=3, smooth=False)
        bleu3 = round(bleu3*100, 2)
    except:
        bleu3 = 0
    try:
        bleu4, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=4, smooth=False)
        bleu4 = round(bleu4*100,2)
    except:
        bleu4 = 0
    
    # rouge score
    score = rouge(predictions, references)
    rouge_s = {k: round(v * 100, 2) for (k, v) in score.items()}
    
    
    # meteor score
    meteor = evaluate.load('meteor')
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
    meteor_score = round(meteor_score*100, 2)    
    
  
    # bert_score
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=predictions, references=references, model_type="microsoft/deberta-xlarge-mnli", lang="en")
    bert_score = round(np.mean(bert_score["f1"])*100,2)
    return {
            "rouge": {"1":rouge_s["rouge_1/f_score"], "2":rouge_s["rouge_2/f_score"], "l":rouge_s["rouge_l/f_score"]},
            "bleu": {"1":bleu1, "2":bleu2, "3":bleu3, "4":bleu4}, 
            "dist": {"1":dist1, "2":dist2},
            "meteor": meteor_score, 
            "bert":bert_score}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def generate_count_mask(tgt_len, device):
    src_len = 3
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    mask[0, 2] = False
    mask[1, 2] = False
    return mask


def generate_peter_mask(tgt_len, device):
    src_len = 2
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    return mask


def generate_square_mask(seqlen, device):
    mask = torch.triu(torch.ones((seqlen, seqlen), device=device), diagonal=1) == 1
    return mask
