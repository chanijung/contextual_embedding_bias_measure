#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
import matplotlib.pyplot as plt
import torch.nn as nn
import time
# get_ipython().run_line_magic('matplotlib', 'inline')

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import sys
sys.path.append("../lib")


# In[2]:


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    model_type="bert-base-uncased",
    max_seq_len=128,
)


# In[3]:


T = TypeVar('T')
def flatten(x: List[List[T]]) -> List[T]:
    return [item for sublist in x for item in sublist]


# In[4]:


from allennlp.common.util import get_spacy_model
from spacy.attrs import ORTH
from spacy.tokenizer import Tokenizer

nlp = get_spacy_model("en_core_web_sm", pos_tags=False, parse=True, ner=False)
nlp.tokenizer.add_special_case("[MASK]", [{ORTH: "[MASK]"}])
def spacy_tok(s: str):
    return [w.text for w in nlp(s)]


def softmax(arr, axis=1):
    e = np.exp(arr)
    return e / e.sum(axis=axis, keepdims=True)


# In[5]:


from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.data.tokenizers import Token

token_indexer = PretrainedBertIndexer(
    pretrained_model=config.model_type,
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )

# apparently we need to truncate the sequence here, which is a stupid design decision
def tokenize(x: str) -> List[Token]:
        return [Token(w) for w in flatten([
                token_indexer.wordpiece_tokenizer(w)
                for w in spacy_tok(x)]
        )[:config.max_seq_len]]




from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params

#BERT
# model = BertForMaskedLM.from_pretrained(config.model_type)

#KnowBERT
archive_file = 'models/kb/wordnet/pure/e1_s240000/model.tar.gz'
print(f'archive: {archive_file}')
params = Params({"archive_file": archive_file})
# load model and batcher
before=time.time()
model = ModelArchiveFromParams.from_params(params=params)
print(f'{time.time()-before} sec taken until model')
# device = list(range(torch.cuda.device_count()))[0]
# model.cuda()
for p in model.parameters():
    p.requires_grad_(False)
print("model built!!!!!!!")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  # model = nn.DataParallel(model)

model.to(device)
model.eval()

batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')
print(f'{time.time()-before} sec taken until batcher')
vocab = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab
print(f'{time.time()-before} sec taken until vocab')
# mask_id = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab['[MASK]']
mask_id = vocab['[MASK]']

# if torch.cuda.is_available():
#     cuda_device = list(range(torch.cuda.device_count()))
#     model = model.cuda(cuda_device[0])
#     print('cuda success')
# else:
#     cuda_device = -1
#     print('cuda fail')



# In[7]:


from allennlp.data import Vocabulary

# vocab = Vocabulary()
# token_indexer._add_encoding_to_vocabulary(vocab)




# In[18]:


def softmax(x, axis=0, eps=1e-9):
    e = np.exp(x)
    return e / (e.sum(axis, keepdims=True) + eps)



def get_logit_scores(input_sentence: str, words: int) -> Dict[str, float]:
    # out_logits = get_logits(input_sentence)
    # input_toks = tokenize(input_sentence)
    # i = _get_mask_index(input_toks)

    sentences = [input_sentence]
    # print(f'sentences: {sentences}')

    # print(f'archive_file {archive_file}')
    # batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')
    # mask_id = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab['[MASK]']
    # print(f'mask_id {mask_id}')

    # batcher takes raw untokenized sentences
    # and yields batches of tensors needed to run KnowBert
    # print(f'created batcher, mask_id {time.time()-start}')
    # device = list(range(torch.cuda.device_count()))[0]
    print(f'Inside get_log_odds')
    start = time.time()
    with torch.no_grad():
        for batch in batcher.iter_batches(sentences):
            print(f'{time.time()-start} sec taken until iter_batches')
            # batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
            # print(f'batch\n{batch}')
            batch_ = move_to(batch, device)
            model_output = model(**batch_)
            # print(f'model_output {time.time()-start}')
            token_mask = batch_['tokens']['tokens'] == mask_id

            # (batch_size, timesteps, vocab_size)
            prediction_scores, _ = model.pretraining_heads(
                    model_output['contextual_embeddings'], model_output['pooled_output']
            )
            # print(f'prediction scores {time.time()-start}')
            # print(f'pred score size {prediction_scores.size()}')

            # (num_masked_tokens, vocab_size)
            mask_token_probabilities = prediction_scores.masked_select(token_mask.unsqueeze(-1)).view(-1, prediction_scores.shape[-1])  # (num_masked_tokens, vocab_size)
            # print(f'mask_token_prob size {mask_token_probabilities.size()}')

    mask_token_probabilities = mask_token_probabilities.detach().cpu().numpy()
    # return {w: out_logits[i, token_indexer.vocab[w]] for w in words}
    
    # print(f'before return {time.time()-start}')
    # for w in words:
        # print(f'vocab index for {w}: {vocab[w]}')
    return {w: mask_token_probabilities[:,vocab[w]] for w in words}

def get_log_odds(input_sentence: str, word1: str, word2: str) -> float:
    scores = get_logit_scores(input_sentence, (word1, word2))
    return scores[word1] - scores[word2]

def get_mask_fill_logits(sentence: str, words: Iterable[str],
                         use_last_mask=False, apply_softmax=False) -> Dict[str, float]:
    # mask_i = processor.get_index(sentence, "[MASK]", last=use_last_mask)
    # logits = defaultdict(list)
    # out_logits = get_logits(sentence)
    # if apply_softmax: 
    #     out_logits = softmax(out_logits)
    # return {w: out_logits[mask_i, processor.token_to_index(w)] for w in words}

    sentences = [sentence]
    print(f'sentences {sentences}')


    # batcher takes raw untokenized sentences
    # and yields batches of tensors needed to run KnowBert
    for batch in batcher.iter_batches(sentences):
        batch_ = move_to(batch, device)
        model_output = model(**batch_)
        
        token_mask = batch_['tokens']['tokens'] == mask_id

        # (batch_size, timesteps, vocab_size)
        prediction_scores, _ = model.pretraining_heads(
                model_output['contextual_embeddings'], model_output['pooled_output']
        )
        print(f'pred score size {prediction_scores.size()}')

        if apply_softmax: 
            prediction_scores = softmax(prediction_scores)

        # (num_masked_tokens, vocab_size)
        mask_token_probabilities = prediction_scores.masked_select(token_mask.unsqueeze(-1)).view(-1, prediction_scores.shape[-1])  # (num_masked_tokens, vocab_size)
        if use_last_mask:
          mask_token_probabilities = mask_token_probabilities[1, :]
        else:
          mask_token_probabilities = mask_token_probabilities[0, :]
        print(f'mask_token_prob size {mask_token_probabilities.size()}')

    for w in words:
        print(f'vocab index for {w}: {vocab[w]}')
    return {w: mask_token_probabilities[vocab[w]] for w in words}


def bias_score(sentence: str, gender_words: Iterable[str], 
               word: str, gender_comes_first=True) -> Dict[str, float]:
    """
    Input a sentence of the form "GGG is XXX"
    XXX is a placeholder for the target word
    GGG is a placeholder for the gendered words (the subject)
    We will predict the bias when filling in the gendered words and 
    filling in the target word.
    
    gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
    """
    # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
    mw, fw = gender_words
    subject_fill_logits = get_mask_fill_logits(
        sentence.replace("XXX", word).replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=not gender_comes_first,
    )
    subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]
    # male words are simply more likely than female words
    # correct for this by masking the target word and measuring the prior probabilities
    subject_fill_prior_logits = get_mask_fill_logits(
        sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"), 
        gender_words, use_last_mask=gender_comes_first,
    )
    subject_fill_bias_prior_correction = subject_fill_prior_logits[mw] - subject_fill_prior_logits[fw]
    
    # probability of filling "programmer" into [MASK] when subject is male/female
    try:
        mw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", mw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        fw_fill_prob = get_mask_fill_logits(
            sentence.replace("GGG", fw).replace("XXX", "[MASK]"), [word],
            apply_softmax=True,
        )[word]
        # We don't need to correct for the prior probability here since the probability
        # should already be conditioned on the presence of the word in question
        tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
    except:
        tgt_fill_bias = np.nan # TODO: handle multi word case
    return {"gender_fill_bias": subject_fill_bias,
            "gender_fill_prior_correction": subject_fill_bias_prior_correction,
            "gender_fill_bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
            "target_fill_bias": tgt_fill_bias, 
           }


def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")

if __name__== '__main__':
  ans = bias_score("GGG is XXX", ["he", "she"], 'good at ')
  score= ans['gender_fill_bias_prior_corrected']
  print(f'score {score}')
  print(f'score.size() {scroe.size()}')
  print(f'score[0] {score[0]}')
