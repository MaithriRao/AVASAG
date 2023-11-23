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

# In[2]:


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

torch.manual_seed(0)
# PyTorch operations must use “deterministic” algorithms. if not available throw RuntimeError
torch.use_deterministic_algorithms(True)

# In[3]:


train_filepaths = ["text_DE.txt", "gloss.txt", "dom_gloss.txt", "ndom_gloss.txt"]

# de --> deutschland \
# gl --> gloss

# In[4]:


import nltk


# nltk.download('punkt')


# In[5]:


def build_vocab(filepath, tokenizer):
    """
    a function to build vocabulary

    :param filepath: file path of the text file
    :param tokenizer: tokenizer related to the text file language
    :return: torchtext vocab of a particular language
    """
    counter = Counter()
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            #             print(string_)
            tokenized_text = tokenizer(string_)
            #             print(tokenized_text)
            counter.update(tokenized_text)
    #             print(counter)
    #             print("***************************")
    #         print("one done!")
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


# In[69]:


from nltk.tokenize import WhitespaceTokenizer

# get spacy tokenizer for German text, get word_tokenzie for gloss text
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
gl_tokenizer = get_tokenizer(word_tokenize)
dom_tokenizer = get_tokenizer(word_tokenize)
ndom_tokenizer = get_tokenizer(word_tokenize)

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
gl_vocab = build_vocab(train_filepaths[1], gl_tokenizer)
dom_vocab = build_vocab(train_filepaths[2], gl_tokenizer)
ndom_vocab = build_vocab(train_filepaths[3], gl_tokenizer)

# In[7]:


print(de_vocab['<unk>'])
print(de_vocab['<pad>'])
print(de_vocab['<bos>'])
print(de_vocab['<eos>'])

# In[8]:


print(gl_vocab['<unk>'])
print(gl_vocab['<pad>'])
print(gl_vocab['<bos>'])
print(gl_vocab['<eos>'])

# I added this two lines because some tokens from the validation and test are not in the train set.

# In[9]:


UNK_IDX = de_vocab['<unk>']
de_vocab.set_default_index(UNK_IDX)
gl_vocab.set_default_index(UNK_IDX)
dom_vocab.set_default_index(UNK_IDX)
ndom_vocab.set_default_index(UNK_IDX)


# In[10]:


def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_gl_iter = iter(io.open(filepaths[1], encoding="utf8"))
    raw_dom_iter = iter(io.open(filepaths[2], encoding="utf8"))
    raw_ndom_iter = iter(io.open(filepaths[3], encoding="utf8"))
    data = []
    for (raw_de, raw_gl, raw_dom, raw_ndom) in zip(raw_de_iter, raw_gl_iter, raw_dom_iter, raw_ndom_iter):
        # rstrip("\n") removes any "/n" from the end of the string
        de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de.rstrip("\n"))],
                                  dtype=torch.long)
        gl_tensor_ = torch.tensor([gl_vocab[token] for token in gl_tokenizer(raw_gl.rstrip("\n"))],
                                  dtype=torch.long)
        dom_tensor_ = torch.tensor([dom_vocab[token] for token in dom_tokenizer(raw_dom.rstrip("\n"))],
                                   dtype=torch.long)
        ndom_tensor_ = torch.tensor([ndom_vocab[token] for token in ndom_tokenizer(raw_ndom.rstrip("\n"))],
                                    dtype=torch.long)
        data.append((de_tensor_, gl_tensor_, dom_tensor_, ndom_tensor_))
    return data


train_data = data_process(train_filepaths)

# In[11]:


txt = "Der Zug nach Hannover fährt auf Gleis 8."
de_tokenizer(txt.rstrip("\n"))

# In[12]:


print(train_data[0])

# In[13]:


print(train_data[1])

# In[14]:


txt = "Die voraussichtliche Ankunfstszeit in Erfurt ist 08:24 Uhr."
de_tokenizer(txt.rstrip("\n"))

# In[15]:


len_train = len(train_data)
print(len(train_data))
# print(len(val_data))
# print(len(test_data))


# In[16]:


print(train_data[0][0])
print(train_data[0][1])
print(train_data[0][2])
print(train_data[0][3])

# In[17]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

# In[18]:


import math

div_1 = math.floor(len(train_data) * 0.50)
div_2 = math.floor(len(train_data) * 0.75)
print(div_1)
print(div_2)

# In[19]:


print(train_data[0:3])

# In[20]:


import random

random.seed(1)
random.shuffle(train_data)

# In[21]:


print(train_data[0:3])

# In[22]:


val_data = train_data[div_1:div_2]
test_data = train_data[div_2:len_train]
train_data = train_data[0:div_1]

# In[23]:


print(len(val_data))
print(len(test_data))
print(len(train_data))

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

# In[24]:


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def generate_batch(data_batch):
    de_batch, gl_batch, dom_batch, ndom_batch = [], [], [], []
    for (de_item, gl_item, dom_item, ndom_item) in data_batch:
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
        gl_batch.append(torch.cat([torch.tensor([BOS_IDX]), gl_item, torch.tensor([EOS_IDX])], dim=0))
        dom_batch.append(torch.cat([torch.tensor([BOS_IDX]), dom_item, torch.tensor([EOS_IDX])], dim=0))
        ndom_batch.append(torch.cat([torch.tensor([BOS_IDX]), ndom_item, torch.tensor([EOS_IDX])], dim=0))
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
    gl_batch = pad_sequence(gl_batch, padding_value=PAD_IDX)
    dom_batch = pad_sequence(dom_batch, padding_value=PAD_IDX)
    ndom_batch = pad_sequence(ndom_batch, padding_value=PAD_IDX)
    return de_batch, gl_batch, dom_batch, ndom_batch


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
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

# In[25]:


from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, gl_vocab_size: int, dom_vocab_size: int, ndom_vocab_size: int,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=NHEAD,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.gl_generator = nn.Linear(emb_size, gl_vocab_size)
        self.dom_generator = nn.Linear(emb_size, dom_vocab_size)
        self.ndom_generator = nn.Linear(emb_size, ndom_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.gl_tok_emb = TokenEmbedding(gl_vocab_size, emb_size)
        self.dom_tok_emb = TokenEmbedding(dom_vocab_size, emb_size)
        self.ndom_tok_emb = TokenEmbedding(ndom_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: Tensor, gl: Tensor, dom: Tensor, ndom: Tensor, src_mask: Tensor,
                gl_mask: Tensor, dom_mask: Tensor, ndom_mask: Tensor, src_padding_mask: Tensor,
                gl_padding_mask: Tensor, dom_padding_mask: Tensor, ndom_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        gl_emb = self.positional_encoding(self.gl_tok_emb(gl))
        dom_emb = self.positional_encoding(self.dom_tok_emb(dom))
        ndom_emb = self.positional_encoding(self.ndom_tok_emb(ndom))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        gl_outs = self.transformer_decoder(gl_emb, memory, gl_mask, None,
                                           gl_padding_mask, memory_key_padding_mask)
        dom_outs = self.transformer_decoder(dom_emb, memory, dom_mask, None,
                                            dom_padding_mask, memory_key_padding_mask)
        ndom_outs = self.transformer_decoder(ndom_emb, memory, ndom_mask, None,
                                             ndom_padding_mask, memory_key_padding_mask)
        return self.gl_generator(gl_outs), self.dom_generator(dom_outs), self.ndom_generator(ndom_outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def gl_decode(self, gl: Tensor, dom: Tensor, ndom: Tensor, memory: Tensor, gl_mask: Tensor, dom_mask: Tensor,
                  ndom_mask: Tensor):
        gl_out = self.transformer_decoder(self.positional_encoding(
            self.gl_tok_emb(gl)), memory,
            gl_mask)
        dom_out = self.transformer_decoder(self.positional_encoding(
            self.dom_tok_emb(dom)), memory,
            dom_mask)
        ndom_out = self.transformer_decoder(self.positional_encoding(
            self.ndom_tok_emb(ndom)), memory,
            ndom_mask)
        return gl_out, dom_out, ndom_out


# Text tokens are represented by using token embeddings. Positional
# encoding is added to the token embedding to introduce a notion of word
# order.
#
#
#

# In[26]:


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
                            self.pos_embedding[:token_embedding.size(0), :])


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

# In[27]:


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, gl, dom, ndom):
    src_seq_len = src.shape[0]
    gl_seq_len = gl.shape[0]
    dom_seq_len = dom.shape[0]
    ndom_seq_len = ndom.shape[0]

    gl_mask = generate_square_subsequent_mask(gl_seq_len)
    dom_mask = generate_square_subsequent_mask(dom_seq_len)
    ndom_mask = generate_square_subsequent_mask(ndom_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    gl_padding_mask = (gl == PAD_IDX).transpose(0, 1)
    dom_padding_mask = (gl == PAD_IDX).transpose(0, 1)
    ndom_padding_mask = (gl == PAD_IDX).transpose(0, 1)
    return src_mask, gl_mask, dom_mask, ndom_mask, src_padding_mask, gl_padding_mask, dom_padding_mask, ndom_padding_mask


# Define model parameters and instantiate model
#
#
#

# In[28]:


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
NUM_EPOCHS = 4

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS,
                                 EMB_SIZE, SRC_VOCAB_SIZE, GL_VOCAB_SIZE, DOM_VOCAB_SIZE, NDOM_VOCAB_SIZE,
                                 FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


# In[29]:


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, gl, dom, ndom) in enumerate(train_iter):
        src = src.to(device)
        gl = gl.to(device)
        dom = dom.to(device)
        ndom = ndom.to(device)

        gl_input = gl[:-1, :]
        dom_input = dom[:-1, :]
        ndom_input = ndom[:-1, :]

        src_mask, gl_mask, dom_mask, ndom_mask, src_padding_mask, gl_padding_mask, dom_padding_mask, ndom_padding_mask = create_mask(
            src, gl_input, dom_input, ndom_input)

        logits1, logits2, logits3 = model(src, gl_input, dom_input, ndom_input, src_mask, gl_mask, dom_mask, ndom_mask,
                                          src_padding_mask, gl_padding_mask, dom_padding_mask, ndom_padding_mask,
                                          src_padding_mask)

        optimizer.zero_grad()

        gl_out = gl[1:, :]
        dom_out = dom[1:, :]
        ndom_out = ndom[1:, :]

        loss1 = loss_fn(logits1.reshape(-1, logits1.shape[-1]), gl_out.reshape(-1))
        loss2 = loss_fn(logits2.reshape(-1, logits2.shape[-1]), dom_out.reshape(-1))
        loss3 = loss_fn(logits3.reshape(-1, logits3.shape[-1]), ndom_out.reshape(-1))

        loss = loss1 + loss2 + loss3

        loss.backward()

        optimizer.step()

        losses += loss.item()
    return losses / len(train_iter)


# In[30]:


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    for idx, (src, gl, dom, ndom) in (enumerate(valid_iter)):
        src = src.to(device)
        gl = gl.to(device)
        dom = dom.to(device)
        ndom = ndom.to(device)

        gl_input = gl[:-1, :]
        dom_input = gl[:-1, :]
        ndom_input = gl[:-1, :]

        src_mask, gl_mask, dom_mask, ndom_mask, src_padding_mask, gl_padding_mask, dom_padding_mask, ndom_padding_mask = create_mask(
            src, gl_input, dom_input, ndom_input)

        logits1, logits2, logits3 = model(src, gl_input, dom_input, ndom_input, src_mask, gl_mask,
                                          dom_mask, ndom_mask, src_padding_mask, gl_padding_mask,
                                          dom_padding_mask, ndom_padding_mask, src_padding_mask)
        gl_out = gl[1:, :]
        loss1 = loss_fn(logits1.reshape(-1, logits1.shape[-1]), gl_out.reshape(-1))
        dom_out = gl[1:, :]
        loss2 = loss_fn(logits2.reshape(-1, logits2.shape[-1]), dom_out.reshape(-1))
        ndom_out = ndom[1:, :]
        loss3 = loss_fn(logits3.reshape(-1, logits3.shape[-1]), ndom_out.reshape(-1))

        loss = loss1 + loss2 + loss3
        losses += loss.item()
    return losses / len(val_iter)


# Train model \
# run this code block 1 time --> the result of decoding is not good \
# run this code block 2 times --> the result of decoding is a bit better, eval loss 1.513, testset wer: 0.6655 \
# run this code block 3 times --> eval loss 0.720, testset wer: 0.6448 \
# run this code block 4 times --> eval loss 0.354, testset wer: 0.6241 \
#
# each run of the code below is 16 epochs,
# the results show overfitting, as the loss of evaluation set is decreasing sharply but the wer for test set is not.

# In[31]:


NUM_EPOCHS = 20

# In[32]:


for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    train_loss = train_epoch(transformer, train_iter, optimizer)
    end_time = time.time()
    val_loss = evaluate(transformer, valid_iter)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
           f"Epoch time = {(end_time - start_time):.3f}s"))


# We get the following results during model training.
#
# ::
#
#        Epoch: 1, Train loss: 5.316, Val loss: 4.065, Epoch time = 35.322s
#        Epoch: 2, Train loss: 3.727, Val loss: 3.285, Epoch time = 36.283s
#        Epoch: 3, Train loss: 3.131, Val loss: 2.881, Epoch time = 37.096s
#        Epoch: 4, Train loss: 2.741, Val loss: 2.625, Epoch time = 37.714s
#        Epoch: 5, Train loss: 2.454, Val loss: 2.428, Epoch time = 38.263s
#        Epoch: 6, Train loss: 2.223, Val loss: 2.291, Epoch time = 38.415s
#        Epoch: 7, Train loss: 2.030, Val loss: 2.191, Epoch time = 38.412s
#        Epoch: 8, Train loss: 1.866, Val loss: 2.104, Epoch time = 38.511s
#        Epoch: 9, Train loss: 1.724, Val loss: 2.044, Epoch time = 38.367s
#        Epoch: 10, Train loss: 1.600, Val loss: 1.994, Epoch time = 38.491s
#        Epoch: 11, Train loss: 1.488, Val loss: 1.969, Epoch time = 38.490s
#        Epoch: 12, Train loss: 1.390, Val loss: 1.929, Epoch time = 38.194s
#        Epoch: 13, Train loss: 1.299, Val loss: 1.898, Epoch time = 38.430s
#        Epoch: 14, Train loss: 1.219, Val loss: 1.885, Epoch time = 38.406s
#        Epoch: 15, Train loss: 1.141, Val loss: 1.890, Epoch time = 38.365s
#        Epoch: 16, Train loss: 1.070, Val loss: 1.873, Epoch time = 38.439s
#
# The models trained using transformer architecture — train faster
# and converge to a lower validation loss compared to RNN models.
#
#

# In[63]:


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    gl_ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    dom_ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    ndom_ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)

    print("max_len", max_len)
    for i in range(max_len - 1):
        #         print(i)
        memory = memory.to(device)
        #         memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        gl_mask = (generate_square_subsequent_mask(gl_ys.size(0))
                   .type(torch.bool)).to(device)
        dom_mask = (generate_square_subsequent_mask(dom_ys.size(0))
                    .type(torch.bool)).to(device)
        ndom_mask = (generate_square_subsequent_mask(ndom_ys.size(0))
                     .type(torch.bool)).to(device)
        out1, out2, out3 = model.gl_decode(gl_ys, dom_ys, ndom_ys, memory, gl_mask, dom_mask, ndom_mask)

        out1 = out1.transpose(0, 1)
        prob1 = model.gl_generator(out1[:, -1])
        _, next_word1 = torch.max(prob1, dim=1)
        next_word1 = next_word1.item()

        gl_ys = torch.cat([gl_ys,
                           torch.ones(1, 1).type_as(src.data).fill_(next_word1)], dim=0)
        if next_word1 == EOS_IDX:
            print("main gloss end generated")
            break

        out2 = out2.transpose(0, 1)
        prob2 = model.dom_generator(out2[:, -1])
        _, next_word2 = torch.max(prob2, dim=1)
        next_word2 = next_word2.item()

        dom_ys = torch.cat([dom_ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word2)], dim=0)
        if next_word2 == EOS_IDX:
            print("dom gloss end generated")
        #             break

        out3 = out3.transpose(0, 1)
        prob3 = model.gl_generator(out3[:, -1])
        _, next_word3 = torch.max(prob3, dim=1)
        next_word3 = next_word3.item()

        ndom_ys = torch.cat([ndom_ys,
                             torch.ones(1, 1).type_as(src.data).fill_(next_word3)], dim=0)
        if next_word3 == EOS_IDX:
            print("ndom gloss end generated")
    #             break

    return gl_ys, dom_ys, ndom_ys


# In[34]:


# 3rd of March edit from here on
# train the model again as you changed it once


# In[74]:


def translate(model, src, src_vocab, gl_vocab, dom_vocab, ndom_vocab, src_tokenizer):
    model.eval()
    tokens = [BOS_IDX] + [src_vocab[tok] for tok in src_tokenizer(src)] + [EOS_IDX]
    num_tokens = len(tokens)
    print("num_tokens: ", num_tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    gl_tokens, dom_tokens, ndom_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5,
                                                       start_symbol=BOS_IDX)
    gl_tokens = gl_tokens.flatten()
    #     print(gl_tokens)
    #     print(dom_tokens)
    #     print(ndom_tokens)
    dom_tokens = dom_tokens.flatten()
    ndom_tokens = ndom_tokens.flatten()
    #     print("tgt_tokens:", tgt_tokens)
    #     .replace("<bos>", "").replace("<eos>", "")
    gl_out = " ".join([gl_vocab.lookup_token(tok) for tok in gl_tokens]).replace("<bos>", "").replace("<eos>", "")
    dom_out = " ".join([dom_vocab.lookup_token(tok) for tok in dom_tokens]).replace("<bos>", "").replace("<eos>", "")
    ndom_out = " ".join([ndom_vocab.lookup_token(tok) for tok in ndom_tokens]).replace("<bos>", "").replace("<eos>", "")
    return gl_out, dom_out, ndom_out


# desired output:\
# GRUND INDEX ZUG DURCHFAHREN2 VORNE-gespiegelt ANDERS ZUG INDEX VERSPAETEN AUFFAHREN EINFLUSS gets:TJA

# In[75]:


translate(transformer, "Grund dafür ist eine Verspätung eines vorausfahrenden Zuges.", de_vocab, gl_vocab, dom_vocab,
          ndom_vocab, de_tokenizer)

# Some checking with the gl tokenizer

# In[71]:


len(de_tokenizer("Grund dafür ist eine Verspätung eines vorausfahrenden Zuges."))

# In[72]:


temp_gl_tokens = gl_tokenizer(
    "GRUND INDEX ZUG DURCHFAHREN2 VORNE-gespiegelt ANDERS ZUG INDEX VERSPAETEN AUFFAHREN EINFLUSS gets:TJA")
print(temp_gl_tokens)
len(temp_gl_tokens)

# In[ ]:


temp_gl_tokens = gl_tokenizer(
    "ZUG KOMMEN(-von-hinten) ICE num:3 num:50 RICHTUNG KOELN HAUPTBAHNHOF DURCHFAHREN(-links) HANNOVER DURCHFAHREN(-links) ESSEN DURCHFAHREN(-links) KOELN INDEX AUFBRECHEN num:13 UHR num:7 num:40")
print(temp_gl_tokens)
len(temp_gl_tokens)

# In[ ]:



