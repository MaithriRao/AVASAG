#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# 
# Language Translation with Transformer
# =====================================
# 
# 
# 

# Data Processing
# ---------------
# 
# torchtext has utilities for creating datasets that can be easily
# iterated through for the purposes of creating a language translation
# model. In this example, we show how to tokenize a raw text sentence,
# build vocabulary, and numericalize tokens into tensor.
# 
# first install spacy using pip. Then download the raw data for the German Spacy tokenizer from
# https://spacy.io/usage/models for tokenizizng German text.
# 
# 

# In[5]:


import math
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torch import Tensor
import io
import time
import os
import pandas as pd
import json

torch.manual_seed(0)
# PyTorch operations must use “deterministic” algorithms. if not available throw RuntimeError
# torch.use_deterministic_algorithms(True)


from datetime import datetime

save_folder_address = str(datetime.now()).replace(" ", "__")


# In[4]:


features_names = ["maingloss", "domgloss", "ndomgloss", "domreloc", "ndomreloc",
                  "domhandrelocx", "domhandrelocy", "domhandrelocz", "domhandrelocax", 
                  "domhandrelocay", "domhandrelocaz", "domhandrelocsx", "domhandrelocsy", "domhandrelocsz",
                  "domhandrotx", "domhandroty", "domhandrotz", 
                  "ndomhandrelocx", "ndomhandrelocy", "ndomhandrelocz", "ndomhandrelocax", 
                  "ndomhandrelocay", "ndomhandrelocaz", "ndomhandrelocsx", "ndomhandrelocsy", "ndomhandrelocsz",
                  "ndomhandrotx", "ndomhandroty", "ndomhandrotz"]

directory = "mms-subset91"
text_directory = "annotations-full/annotations"
data_list = []
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    df = pd.read_csv(f)
    
    filenumber = filename.split(".")[0]
    
    text_address = os.path.join(text_directory, filenumber, "gebaerdler.Text_Deutsch.annotation~")
    file = open(text_address, encoding='latin-1')
    lines = file.readlines()
    text_line = ""
    for i, text_data in enumerate(lines):
        if i>0:
            text_line = text_line + " " + text_data.replace("\n", "").split(";")[2] 
        else:
            text_line = text_line + text_data.replace("\n", "").split(";")[2] 
    
    data_dict = {"file_ID":filenumber, "text": text_line}
    for feature in features_names:
        if feature == "domgloss" or feature == "ndomgloss":
            temp = df[feature].copy()
            data_dict[feature] = [data_dict["maingloss"][i] if pd.isnull(token) else token for i,token in enumerate(temp)]
        else:
            data_dict[feature] = df[feature].tolist()
    data_list.append(data_dict)


# In[5]:


# len(data_list)


# data_list is a list of dictionaries\
# each dictianry corresponds to a data sample in the dataset\
# file_ID is the file number, text is the german sentence, and the rest are all a list of the same length containing different values of gloss, boolean, and real value numbers.

# In[6]:


boolean_map = {"yes": 1, "no": 0}
for data in data_list:
    data["domreloc"] = [boolean_map[value] for value in data["domreloc"]]
    data["ndomreloc"] = [boolean_map[value] for value in data["ndomreloc"]]
#     data["shoulders"] = [boolean_map[value] for value in data["shoulders"]]


# In[7]:


def build_German_vocab(data_list, tokenizer):
    """
    a function to build vocabulary

    :param filepath: file path of the text file
    :param tokenizer: tokenizer related to the text file language
    :return: torchtext vocab of a particular language
    """ 
    counter = Counter()
    for data in data_list:
        tokenized_text = tokenizer(data["text"])
        counter.update(tokenized_text)
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


# In[8]:


def build_gloss_vocab(data_list, gloss_name):
    """
    a function to build vocabulary

    :param filepath: file path of the text file
    :param tokenizer: tokenizer related to the text file language
    :return: torchtext vocab of a particular language
    """ 
    counter = Counter()
    for data in data_list:
        counter.update(data[gloss_name])
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


# In[9]:


# get spacy tokenizer for German text
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

de_vocab = build_German_vocab(data_list, de_tokenizer)
gl_vocab = build_gloss_vocab(data_list, "maingloss")
dom_vocab = build_gloss_vocab(data_list, "domgloss")
ndom_vocab = build_gloss_vocab(data_list, "ndomgloss")


# In[10]:


# print(de_vocab['<unk>'])
# print(de_vocab['<pad>'])
# print(de_vocab['<bos>'])
# print(de_vocab['<eos>'])


# I added this two lines because some tokens from the validation and test are not in the train set.

# In[11]:


UNK_IDX = de_vocab['<unk>']
de_vocab.set_default_index(UNK_IDX)
gl_vocab.set_default_index(UNK_IDX)
dom_vocab.set_default_index(UNK_IDX)
ndom_vocab.set_default_index(UNK_IDX)


# In[12]:


# # divide train and test here
# import math
# import random

# div = math.floor(len(data_list)*0.75)

# data_list_copy = data_list.copy()
# random.seed(1)
# random.shuffle(data_list_copy)

# test_data_raw = data_list_copy[div:len(data_list)]
# train_data_raw = data_list_copy[0:div]


# In[6]:


with open('test_data.json', 'r') as openfile:
    test_data_raw = json.load(openfile)
    
print(len(test_data_raw))


with open('train_data.json', 'r') as openfile:
    train_data_raw = json.load(openfile)
    
print(len(train_data_raw))


# In[13]:


# print(len(test_data_raw))
# for d in test_data_raw:
#     print(d["file_ID"])


# In[14]:


# print(train_data_raw[0])


# In[15]:


def data_process(data_list):
    processed_data = []
    for data in data_list:
        res_list = []
#         print(data["file_ID"])
        # rstrip("\n") removes any "/n" from the end of the string
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(data["text"].rstrip("\n"))],
                                  dtype=torch.long)
        res_list.append(de_tensor_)
        
        gl_tensor_ = torch.tensor([gl_vocab[token] for token in data["maingloss"]], dtype=torch.long)
        res_list.append(gl_tensor_)
        
        dom_tensor_ = torch.tensor([dom_vocab[token] for token in data["domgloss"]], dtype=torch.long)
        res_list.append(dom_tensor_)
        
        ndom_tensor_ = torch.tensor([ndom_vocab[token] for token in data["ndomgloss"]], dtype=torch.long)
        res_list.append(ndom_tensor_)
        
        for i in range(3, len(features_names)):
            res_list.append(torch.tensor(data[features_names[i]], dtype=torch.float))
            
        processed_data.append(tuple(res_list))
    return processed_data


train_data = data_process(train_data_raw)
test_data = data_process(test_data_raw)


# In[16]:


# print(len(train_data))
# print(train_data[0])


# In[17]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']


# In[18]:


# print(PAD_IDX)
# print(BOS_IDX)
# print(EOS_IDX)


# DataLoader
# ----------
# 
# The last torch specific feature we’ll use is the DataLoader, which is
# easy to use since it takes the data as its first argument. Specifically,
# as the docs say: DataLoader combines a dataset and a sampler, and
# provides an iterable over the given dataset. The DataLoader supports
# both map-style and iterable-style datasets with single- or multi-process
# loading, customizing loading order and optional automatic batching
# (collation) and memory pinning.
# 
# Please pay attention to collate_fn (optional) that merges a list of
# samples to form a mini-batch of Tensor(s). Used when using batched
# loading from a map-style dataset.
# 
# 
# 

# In[19]:


input_names = ["text"]
for name in features_names:
    input_names.append(name)
# input_names


# In[20]:


# len(input_names)


# In[21]:


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def generate_batch(data_batch):
    batches_list = [[] for feature in input_names]
    for items_list in data_batch:
        
        for i, item in enumerate(items_list):
            batches_list[i].append(item)
            
    for i, batch in enumerate(batches_list):
        batches_list[i] = pad_sequence(batch, padding_value=PAD_IDX)
    
    return batches_list

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


# Transformer!
# ------------
# 
# Transformer is a Seq2Seq model introduced in `“Attention is all you
# need” <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# paper for solving machine translation task. Transformer model consists
# of an encoder and decoder block each containing fixed number of layers.
# 
# Encoder processes the input sequence by propogating it, through a series
# of Multi-head Attention and Feed forward network layers. The output from
# the Encoder referred to as ``memory``, is fed to the decoder along with
# target tensors. Encoder and decoder are trained in an end-to-end fashion
# using teacher forcing technique.
# 
# 
# 

# In[22]:


from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, gl_vocab_size: int, dom_vocab_size: int, ndom_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        # real value generation (24 items)
        
        self.linears_real = nn.ModuleList([nn.Linear(emb_size, 1) for i in range(0,24)])
        
        # boolean value generation (2 items)
        self.linears_boolean = nn.ModuleList([nn.Linear(emb_size, 2) for i in range(0,2)])
        
        # text generation
        self.gl_generator = nn.Linear(emb_size, gl_vocab_size)
        self.dom_generator = nn.Linear(emb_size, dom_vocab_size)
        self.ndom_generator = nn.Linear(emb_size, ndom_vocab_size)
        
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.gl_tok_emb = TokenEmbedding(gl_vocab_size, emb_size)
        self.dom_tok_emb = TokenEmbedding(dom_vocab_size, emb_size)
        self.ndom_tok_emb = TokenEmbedding(ndom_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, gl: Tensor, src_mask: Tensor,
                gl_mask: Tensor, src_padding_mask: Tensor,
                gl_padding_mask: Tensor, memory_key_padding_mask: Tensor):
#         dom: Tensor, ndom: Tensor, dom_mask: Tensor, ndom_mask: Tensor, dom_padding_mask: Tensor, ndom_padding_mask: Tensor,
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        gl_emb = self.positional_encoding(self.gl_tok_emb(gl))
#         dom_emb = self.positional_encoding(self.dom_tok_emb(dom))
#         ndom_emb = self.positional_encoding(self.ndom_tok_emb(ndom))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        gl_outs = self.transformer_decoder(gl_emb, memory, gl_mask, None,
                                        gl_padding_mask, memory_key_padding_mask)
#         dom_outs = self.transformer_decoder(dom_emb, memory, dom_mask, None,
#                                         dom_padding_mask, memory_key_padding_mask)
#         ndom_outs = self.transformer_decoder(ndom_emb, memory, ndom_mask, None,
#                                         ndom_padding_mask, memory_key_padding_mask)
        return_list = []
        return_list.append(self.gl_generator(gl_outs))
        return_list.append(self.dom_generator(gl_outs))
        return_list.append(self.ndom_generator(gl_outs))
        
        for bool_layer in self.linears_boolean:
            return_list.append(bool_layer(gl_outs))
        
        for real_layer in self.linears_real:
            return_list.append(real_layer(gl_outs))
            
        return return_list

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def gl_decode(self, gl: Tensor, memory: Tensor, gl_mask: Tensor):
        gl_out = self.transformer_decoder(self.positional_encoding(
                          self.gl_tok_emb(gl)), memory,
                          gl_mask)
        return gl_out


# Text tokens are represented by using token embeddings. Positional
# encoding is added to the token embedding to introduce a notion of word
# order.
# 
# 
# 

# In[23]:


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# We create a ``subsequent word`` mask to stop a target word from
# attending to its subsequent words. We also create masks, for masking
# source and target padding tokens
# 
# 
# 

# In[24]:


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, gl):
    src_seq_len = src.shape[0]
    gl_seq_len = gl.shape[0]
    
    gl_mask = generate_square_subsequent_mask(gl_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    gl_padding_mask = (gl == PAD_IDX).transpose(0, 1)
    return src_mask, gl_mask, src_padding_mask, gl_padding_mask


# Define model parameters and instantiate model 
# 
# 
# 

# In[25]:


# len(gl_vocab)


# In[26]:


SRC_VOCAB_SIZE = len(de_vocab)
GL_VOCAB_SIZE = len(gl_vocab)
DOM_VOCAB_SIZE = len(dom_vocab)
NDOM_VOCAB_SIZE = len(ndom_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, GL_VOCAB_SIZE, DOM_VOCAB_SIZE, NDOM_VOCAB_SIZE, 
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
loss_binary_fn = torch.nn.BCELoss()
loss_real_values = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-09
)


# In[27]:


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, data_tuple in enumerate(train_iter):
        data_list = [item.to(device) for item in data_tuple]
        
        src = data_list[0] 
        gl = data_list[1]
        gl_input = gl[:-1, :]
        
        
        # remove these later
        
        src_mask, gl_mask, src_padding_mask, gl_padding_mask = create_mask(src, gl_input)
        

        logits_list = model(src, gl_input, src_mask, gl_mask,
                            src_padding_mask, gl_padding_mask, src_padding_mask)
#         dom_input, ndom_input, dom_mask, ndom_mask, dom_padding_mask, ndom_padding_mask,
        optimizer.zero_grad()
        
        data_out_list = [data_item[1:,:] for data_item in data_list[1:]]
        
        ### 3 gloss, 2 boolean, 24 real values
        
        for i in range(0,3):
            loss = loss_fn(logits_list[i].reshape(-1, logits_list[i].shape[-1]), data_out_list[i].reshape(-1))
            losses += loss
        
        for i in range(3, 5):
            _, temp1 = torch.max(logits_list[i].reshape(-1, logits_list[i].shape[-1]), dim = 1)
            temp2 = data_out_list[i].reshape(-1)
            
            temp1 = temp1.type('torch.FloatTensor')
            temp1 = temp1.to(device)
            
            loss = loss_binary_fn(temp1, temp2)
            losses += loss
        
        for i in range(5, 29):
            loss = loss_real_values(logits_list[i].reshape(-1, logits_list[i].shape[-1]), data_out_list[i].reshape(-1))
            losses += loss
            


#         loss1 = loss_fn(logits1.reshape(-1, logits1.shape[-1]), gl_out.reshape(-1))
#         loss2 = loss_fn(logits2.reshape(-1, logits2.shape[-1]), dom_out.reshape(-1))
#         loss3 = loss_fn(logits3.reshape(-1, logits3.shape[-1]), ndom_out.reshape(-1))
#         loss4 = loss_binary_fn(temp1, temp2)
#         loss5 = loss_real_values(logits5.reshape(-1, logits5.shape[-1]), domhandrelocx_out.reshape(-1))
        
#         loss = loss1+loss2+loss3+loss4+loss5
        
        # ADD LATER: normalize the loss before backpropagation
        losses.backward()
        
        optimizer.step()
        
        
    return losses / len(train_iter)


# In[28]:


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, data_tuple in enumerate(valid_iter):
        data_list = [item.to(device) for item in data_tuple]
        
        src = data_list[0] 
        gl = data_list[1]
        gl_input = gl[:-1, :]
        
        
        src_mask, gl_mask, src_padding_mask, gl_padding_mask = create_mask(src, gl_input)
        

        logits_list = model(src, gl_input, src_mask, gl_mask,
                            src_padding_mask, gl_padding_mask, src_padding_mask)

        
        data_out_list = [data_item[1:,:] for data_item in data_list[1:]]
        
        
        for i in range(0,3):
            loss = loss_fn(logits_list[i].reshape(-1, logits_list[i].shape[-1]), data_out_list[i].reshape(-1))
            losses += loss
        
        for i in range(3, 5):
            _, temp1 = torch.max(logits_list[i].reshape(-1, logits_list[i].shape[-1]), dim = 1)
            temp2 = data_out_list[i].reshape(-1)
            
            temp1 = temp1.type('torch.FloatTensor')
            
            loss = loss_binary_fn(temp1, temp2)
            losses += loss
        
        for i in range(5, 29):
            loss = loss_real_values(logits_list[i].reshape(-1, logits_list[i].shape[-1]), data_out_list[i].reshape(-1))
            losses += loss
            
    return losses / len(val_iter)


# In[29]:


NUM_EPOCHS = 1000
loss_graf = []


# In[30]:


train_log = open(save_folder_address+"_train_log.txt", 'w')
best_epoch = 0

for epoch in range(1, NUM_EPOCHS+1):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer)
    train_loss = train_loss.tolist()
    end_time = time.time()
    log = "Epoch: " + str(epoch)+", Train loss: "+ str(train_loss)+" Epoch duration "+ str(end_time - start_time)+"\n"
    train_log.write(log)
    if epoch>1 and train_loss < min(loss_graf):
        torch.save(transformer.state_dict(), save_folder_address+"_best_model.pt")
        log = "min so far is at epoch: "+ str(epoch)+"\n"
        train_log.write(log)
        best_epoch = epoch
        
    loss_graf.append(train_loss)
#     val_loss = evaluate(transformer, valid_iter)
#     print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
#            f"Epoch time = {(end_time - start_time):.3f}s"))
log = "best epoch is: "+ str(best_epoch)
train_log.write(log)
train_log.close()


# In[31]:


import matplotlib.pyplot as plt
epochs = [i for i in range(1,NUM_EPOCHS+1)]
plt.plot(epochs, loss_graf)
plt.savefig(save_folder_address+ '_loss_over_'+str(NUM_EPOCHS)+' epochs.png')


# In[32]:


torch.save(transformer.state_dict(), save_folder_address+"_last_model.pt")


# In[33]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    memory = model.encode(src, src_mask)
    
    ys_list = [torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device) for i in range(0,5)]
    for i in range(5, 29):
        ys_list.append(torch.ones(1, 1).fill_(start_symbol).type(torch.float).to(device))
    
    for i in range(max_len-1):
        memory = memory.to(device)
        gl_mask = (generate_square_subsequent_mask(ys_list[0].size(0))
                                    .type(torch.bool)).to(device)
        out = model.gl_decode(ys_list[0],  memory, gl_mask)
        
        out = out.transpose(0, 1)
        
        prob1 = model.gl_generator(out[:, -1])
        _, next_gloss = torch.max(prob1, dim = 1)
        next_gloss = next_gloss.item()
        
        ys_list[0] = torch.cat([ys_list[0],
                            torch.ones(1, 1).type(torch.long).fill_(next_gloss).to(device)], dim=0)
        
        
        if next_gloss == EOS_IDX:
#             print("main gloss end generated")
            break
        
        # dom
        
        prob_dom = model.dom_generator(out[:, -1])
        _, next_dom = torch.max(prob_dom, dim = 1)
        next_dom = next_dom.item()

        ys_list[1] = torch.cat([ys_list[1],
                            torch.ones(1, 1).type(torch.long).fill_(next_dom).to(device)], dim=0)
        
        # ndom
        prob_ndom = model.ndom_generator(out[:, -1])
        _, next_ndom = torch.max(prob_ndom, dim = 1)
        next_ndom = next_ndom.item()

        ys_list[2] = torch.cat([ys_list[2],
                            torch.ones(1, 1).type(torch.long).fill_(next_ndom).to(device)], dim=0)
        
        # boolean
        for i in range(3,5):
            prob = model.linears_boolean[i-3](out[:, -1])
            _, next_bool = torch.max(prob, dim = 1)
            next_bool = next_bool.item()
            
            ys_list[i] = torch.cat([ys_list[i],
                            torch.ones(1, 1).type(torch.long).fill_(next_bool).to(device)], dim=0)

        # real values
        for i in range(5,29):
            next_real = model.linears_real[i-6](out[:, -1])
            next_real = next_real.item()

#             print(next_real)
            ys_list[i] = torch.cat([ys_list[i],
                            torch.ones(1, 1).type(torch.float).fill_(next_real).to(device)], dim=0)
#             print(ys_list[i])
        
    return ys_list


# In[34]:


def translate(model, src, src_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    ys_list  = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX)
    return ys_list


# let's get some numbers!
# testing the model!

# In[35]:


eval_lists_dict = {}

for feature in features_names:
    eval_lists_dict[feature] = {"true": [], "pred": []}

# number of times that length of predecited sequence is higher than the true sequence
num_P_T = 0

# number of times that length of predecited sequence is lower than the true sequence
num_T_P = 0

# number of times that length of predecited sequence is same as the true sequence
num_e = 0

for entry in test_data_raw:
#     print("*************************************************")
#     print(entry)
#     print("*************************************************")
    
    de_text = entry["text"]
    gl_text = " ".join(entry["maingloss"])
    
    # get model predictions for the de_text as input
    ys_list = translate(transformer, de_text, de_vocab, de_tokenizer)
    
    # maingloss
    gl_tokens = ys_list[0].flatten()
    gl_pred = " ".join([gl_vocab.lookup_token(tok) for tok in gl_tokens]).replace("<bos>", "").replace("<eos>", "")
    
    eval_lists_dict["maingloss"]["true"].append(gl_text)
    eval_lists_dict["maingloss"]["pred"].append(gl_pred)
    
    # domgloss
    dom_text = " ".join(entry["domgloss"])
    dom_tokens = ys_list[1].flatten()
    dom_pred = " ".join([dom_vocab.lookup_token(tok) for tok in dom_tokens]).replace("<bos>", "").replace("<eos>", "")
    
    eval_lists_dict["domgloss"]["true"].append(dom_text)
    eval_lists_dict["domgloss"]["pred"].append(dom_pred)
    
    
    # ndom
    ndom_text = " ".join(entry["ndomgloss"])
    ndom_tokens = ys_list[2].flatten()
    ndom_pred = " ".join([ndom_vocab.lookup_token(tok) for tok in ndom_tokens]).replace("<bos>", "").replace("<eos>", "")
    
    eval_lists_dict["ndomgloss"]["true"].append(ndom_text)
    eval_lists_dict["ndomgloss"]["pred"].append(ndom_pred)
    
    
    
    # domreloc
    dom_reloc_true = entry["domreloc"]
    dom_reloc_pred = ys_list[3].flatten()
        
    dom_reloc_true = dom_reloc_true
    dom_reloc_pred = dom_reloc_pred.tolist()
    dom_reloc_pred = dom_reloc_pred[1:]
        
    P = len(dom_reloc_pred)
    T = len(dom_reloc_true)
            
    if P>T:
        num_P_T = num_P_T+1
        dom_reloc_pred = dom_reloc_pred[:T]
    elif T>P:
        num_T_P = num_T_P+1
        dom_reloc_true = dom_reloc_true[:P]
    else:
        num_e = num_e+1
    
    eval_lists_dict["domreloc"]["true"].append(dom_reloc_true)
    eval_lists_dict["domreloc"]["pred"].append(dom_reloc_pred)
    
    
    # binary and real values
    for i, param in enumerate(features_names[4:]):
        true = entry[features_names[i+4]]
        pred = ys_list[i+4].flatten()
        
        true = true
        pred = pred.tolist()
        pred = pred[1:]
        
        if P>T:
            pred = pred[:T]
        elif T>P:
            true = true[:P]
            
        
        eval_lists_dict[features_names[i+4]]["true"].append(true)
        eval_lists_dict[features_names[i+4]]["pred"].append(pred)
    


# In[36]:


# print(len(eval_lists_dict["maingloss"]["true"]))
# print(len(eval_lists_dict["maingloss"]["pred"]))

f = open(save_folder_address+"_outputs.txt","w")

line = "P>T: "+ str(num_P_T) +"\n"
f.write(line)

line = "T>P: "+ str(num_T_P) +"\n"
f.write(line)

line = "equal: "+ str(num_e) +"\n"
f.write(line)


# In[37]:


# len(features_names)


# In[38]:


from sacrebleu.metrics import BLEU

# use the lists ground_truth, hypothesis


bleu = BLEU()

result = bleu.corpus_score(eval_lists_dict["maingloss"]["pred"], eval_lists_dict["maingloss"]["true"])
line = "BLEU score for maingloss: "+str(result)+"\n"
f.write(line)

result = bleu.corpus_score(eval_lists_dict["domgloss"]["pred"], eval_lists_dict["domgloss"]["true"])
line = "BLEU score for domgloss: "+str(result)+"\n"
f.write(line)

result = bleu.corpus_score(eval_lists_dict["ndomgloss"]["pred"], eval_lists_dict["ndomgloss"]["true"])
line = "BLEU score for ndomgloss: "+str(result)+"\n"
f.write(line)


# In[39]:


print(len(gl_vocab))
print(len(dom_vocab))
print(len(ndom_vocab))


# In[40]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


for feature in features_names[3:5]:
    temp = []
    for true, pred in zip(eval_lists_dict[feature]["true"], eval_lists_dict[feature]["pred"]):
        temp.append(accuracy_score(true, pred))
    line = str(feature)+ " : " + str(sum(temp)/len(temp))+"\n"
    f.write(line)
    
for feature in features_names[5:]:
    temp = []
    for true, pred in zip(eval_lists_dict[feature]["true"], eval_lists_dict[feature]["pred"]):
        temp.append(mean_squared_error(true, pred))
    line = str(feature)+ " : " + str(sum(temp)/len(temp))+"\n"
    f.write(line)


# In[41]:


f.close()

