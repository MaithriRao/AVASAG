#!/usr/bin/env python
# coding: utf-8

# In[64]:


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
from datetime import datetime

torch.manual_seed(0)

save_folder_address = "onlyGloss/"+str(datetime.now()).replace(" ", "__")



# In[65]:


features_names = ["maingloss"]

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
        data_dict[feature] = df[feature].tolist()
    data_list.append(data_dict)


# In[66]:


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


# In[67]:


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


# In[68]:


# get spacy tokenizer for German text
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')


de_vocab = build_German_vocab(data_list, de_tokenizer)
gl_vocab = build_gloss_vocab(data_list, "maingloss")


# In[69]:


UNK_IDX = de_vocab['<unk>']
de_vocab.set_default_index(UNK_IDX)
gl_vocab.set_default_index(UNK_IDX)


# In[70]:


with open('test_data.json', 'r') as openfile:
    test_data_raw = json.load(openfile)
    
print(len(test_data_raw))


with open('train_data.json', 'r') as openfile:
    train_data_raw = json.load(openfile)
    
print(len(train_data_raw))


# In[71]:


def data_process(data_list):
    processed_data = []
    for data in data_list:
        res_list = []
        # rstrip("\n") removes any "/n" from the end of the string
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(data["text"].rstrip("\n"))],
                                  dtype=torch.long)
        res_list.append(de_tensor_)
        
        gl_tensor_ = torch.tensor([gl_vocab[token] for token in data["maingloss"]], dtype=torch.long)
        res_list.append(gl_tensor_)
            
        processed_data.append(tuple(res_list))
    return processed_data


train_data = data_process(train_data_raw)
test_data = data_process(test_data_raw)


# In[72]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']


# In[73]:


input_names = ["text"]
for name in features_names:
    input_names.append(name)
input_names


# In[97]:


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
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)


# In[98]:


from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
                
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor,
                tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


# In[99]:


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


# In[100]:


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


# In[101]:


SRC_VOCAB_SIZE = len(de_vocab)
TGT_VOCAB_SIZE = len(gl_vocab)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, 
                                 EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


# In[102]:


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:-1, :]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        losses += loss
        
        losses.backward()

        optimizer.step()
        

    return losses / len(train_iter)




# In[103]:


NUM_EPOCHS = 1000
loss_graf = []


# In[104]:


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
    
log = "best epoch is: "+ str(best_epoch)
train_log.write(log)
train_log.close()


# In[105]:


import matplotlib.pyplot as plt
epochs = [i for i in range(1,NUM_EPOCHS+1)]
plt.plot(epochs, loss_graf)
plt.savefig(save_folder_address+ '_loss_over_'+str(NUM_EPOCHS)+' epochs.png')


# In[106]:


torch.save(transformer.state_dict(), save_folder_address+"_last_model.pt")


# In[107]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# In[111]:


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    ys_list = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return ys_list


# In[112]:


translate(transformer, "Heute kein Halt in Köln.", de_vocab, gl_vocab, de_tokenizer)


# expected:
# HEUTE
# DORTHIN
# KOELN
# ANHALTEN
# NICHT

# In[133]:


ground_truth = []
hypothesis = []

# number of times that length of predecited sequence is higher than the true sequence
num_P_T = 0

# number of times that length of predecited sequence is lower than the true sequence
num_T_P = 0

# number of times that length of predecited sequence is same as the true sequence
num_e = 0

for entry in test_data_raw:
    
    de_text = entry['text']
    gl_text = " ".join(entry["maingloss"])
    
    ys_list = translate(transformer, de_text, de_vocab, gl_vocab, de_tokenizer)
    
    gl_tokens = ys_list.flatten()

    gl_pred = " ".join([gl_vocab.lookup_token(tok) for tok in gl_tokens]).replace("<bos>", "").replace("<eos>", "")

    ground_truth.append(gl_text)
    hypothesis.append(gl_pred)
    
    P = len(gl_tokens.tolist())-1
    T = len(entry["maingloss"]) 
    if P>T:
        num_P_T = num_P_T+1
    elif T>P:
        num_T_P = num_T_P+1
    else:
        num_e = num_e+1
    


# In[135]:


f = open(save_folder_address+"_outputs.txt","w")

line = "P>T: "+ str(num_P_T) +"\n"
f.write(line)

line = "T>P: "+ str(num_T_P) +"\n"
f.write(line)

line = "equal: "+ str(num_e) +"\n"
f.write(line)


# In[136]:


from sacrebleu.metrics import BLEU

# use the lists ground_truth, hypothesis
refs = [ground_truth]

bleu = BLEU()

result = bleu.corpus_score(hypothesis, refs)

line = "BLEU score for maingloss: "+str(result)+"\n"
f.write(line)


# In[137]:


f.close()


# In[ ]:




