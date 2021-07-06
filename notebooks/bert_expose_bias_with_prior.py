#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


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


# In[6]:


from pytorch_pretrained_bert import BertConfig, BertForMaskedLM
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params

#BERT
# model = BertForMaskedLM.from_pretrained(config.model_type)

#KnowBERT
archive_file = 'debiased_wordnet/model.tar.gz'
params = Params({"archive_file": archive_file})
# load model and batcher
model = ModelArchiveFromParams.from_params(params=params)
model.cuda()
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
print("model built!!!!!!!")
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


# In[8]:


# def get_logits(input_sentence: str) -> torch.Tensor:
#     input_toks = tokenize(input_sentence)
#     batch = token_indexer.tokens_to_indices(input_toks, vocab, "tokens")
#     token_ids = torch.LongTensor(batch["tokens"]).unsqueeze(0)
#     with torch.no_grad():
#         out_logits = model(token_ids).squeeze(0)
#     return out_logits.detach().cpu().numpy()


# In[9]:


# full_vocab = {v:k for k, v in token_indexer.vocab.items()}

# def indices_to_words(indices: Iterable[int]) -> List[str]:
#     return [full_vocab[x] for x in indices]







# Measuring difference

# In[17]:


# male_logits = get_logits("he is very [MASK].")[4, :]
# female_logits = get_logits("she is very [MASK].")[4, :]


# In[18]:


def softmax(x, axis=0, eps=1e-9):
    e = np.exp(x)
    return e / (e.sum(axis, keepdims=True) + eps)


# In[19]:


# male_probs = softmax(male_logits)
# female_probs = softmax(female_logits)


# In[20]:


# msk = ((male_probs >= 1e-6) & (female_probs >= 1e-6))
# male_probs = male_probs[msk]
# female_probs = female_probs[msk]


# In[21]:


# [(pos + 1, full_vocab[i]) for i, pos in enumerate((male_probs / female_probs).argsort()) if pos < 10]


# In[22]:


# [(pos + 1, full_vocab[i]) for i, pos in enumerate((female_probs / male_probs).argsort()) if pos < 10]


# In[ ]:





# # Construct measure of bias

# In[23]:


# input_sentence = "[MASK] is intelligent"


# In[24]:


# def _get_mask_index(toks: Iterable[Token]) -> int:
#     for i, t in enumerate(toks):
#         if t.text == "[MASK]":
#             return i + 1 # take the [CLS] token into account
#     raise ValueError("No [MASK] token found")


# In[25]:


# def get_logits(input_sentence: str, n_calc: int=10) -> np.ndarray:
#     """
#     n_calc: Since the logits are non-deterministic, 
#     computing the logits multiple times might be better
#     """
#     input_toks = tokenize(input_sentence)
#     batch = token_indexer.tokens_to_indices(input_toks, vocab, "tokens")
#     token_ids = torch.LongTensor(batch["tokens"]).unsqueeze(0)
    
#     logits = None
#     for _ in range(n_calc):
#         with torch.no_grad():
#             out_logits = model(token_ids).squeeze(0)
#         if logits is None: logits = np.zeros(out_logits.shape)
#         logits += out_logits.detach().cpu().numpy()
#     return logits / n_calc


# In[26]:


def get_logit_scores(input_sentence: str, words: int) -> Dict[str, float]:
    # out_logits = get_logits(input_sentence)
    # input_toks = tokenize(input_sentence)
    # i = _get_mask_index(input_toks)

    sentences = [input_sentence]
    print(f'sentences: {sentences}')

    printf(f'archive_file {archive_file}')
    batcher = KnowBertBatchifier(archive_file, masking_strategy='full_mask')
    mask_id = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab['[MASK]']
    print(f'mask_id {mask_id}')

    # batcher takes raw untokenized sentences
    # and yields batches of tensors needed to run KnowBert
    with torch.no_grad():
        for batch in batcher.iter_batches(sentences):
            model_output = model(**batch)
            token_mask = batch['tokens']['tokens'] == mask_id

            # (batch_size, timesteps, vocab_size)
            prediction_scores, _ = model.pretraining_heads(
                    model_output['contextual_embeddings'], model_output['pooled_output']
            )
            print(f'pred score size {prediction_scores.size()}')

            # (num_masked_tokens, vocab_size)
            mask_token_probabilities = prediction_scores.masked_select(token_mask.unsqueeze(-1)).view(-1, prediction_scores.shape[-1])  # (num_masked_tokens, vocab_size)
            print(f'mask_token_prob size {mask_token_probabilities.size()}')

    mask_token_probabilities = mask_token_probabilities.detach().cpu().numpy()
    # return {w: out_logits[i, token_indexer.vocab[w]] for w in words}
    vocab = batcher.tokenizer_and_candidate_generator.bert_tokenizer.vocab
    for w in words:
        print(f'vocab index for {w}: {vocab[w]}')
    return {w: mask_token_probabilities[:,vocab[w]] for w in words}

def get_log_odds(input_sentence: str, word1: str, word2: str) -> float:
    scores = get_logit_scores(input_sentence, (word1, word2))
    return scores[word1] - scores[word2]




