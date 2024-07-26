import torch
import torch.nn as nn
import math
import pickle
import os
import io
import time
import pandas as pd
import json
from pathlib import Path
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)
from .utils import Tokenization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model:
    def __init__(self, datasets, augment):
        self.augment = augment
        (original, modified, full) = datasets
        if self.augment:
            (tokenizer, vocab, sentences) = full
        else:
            (tokenizer, vocab, sentences) = original

        (self.text_tokenizer, self.mms_tokenizer) = tokenizer
        (self.text_vocab, self.mms_vocab) = vocab

        self.UNK_IDX = self.text_vocab['<unk>']
        self.text_vocab.set_default_index(self.UNK_IDX)
        self.mms_vocab.set_default_index(self.UNK_IDX)

        self.PAD_IDX = self.text_vocab['<pad>']
        self.BOS_IDX = self.text_vocab['<bos>']
        self.EOS_IDX = self.text_vocab['<eos>']
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)


    def data_process(self, source_sentences, target_sentences, tokenization):
        data = []
        for ((text_id, text), (mms_id, mms)) in zip(source_sentences, target_sentences):
            text_tokens = self.text_tokenizer.encode(text.rstrip("\n"), out_type=str)
            text_tensor_ = torch.tensor([self.text_vocab[token] for token in self.text_tokenizer.encode(text.rstrip("\n"), out_type=str)], dtype=torch.long) 
            if tokenization == Tokenization.SOURCE_TARGET: 
                mms_tensor_ = torch.tensor([self.mms_vocab[token] for token in self.mms_tokenizer.encode(mms.rstrip("\n"), out_type=str)], dtype=torch.long)
            elif tokenization == Tokenization.SOURCE_ONLY:
                tokenized_mms = []
                for token in mms:
                    tokenized_mms.append(self.mms_vocab[token])
                mms_tensor_ = torch.tensor(tokenized_mms, dtype=torch.long)
            else:
                raise ValueError("Invalid tokenization mode")
            data.append((text_tensor_, mms_tensor_))
        return data


    def generate_batch(self, data_batch):
        text_batch, mms_batch = [], []
        for (text_item, mms_item) in data_batch:
            text_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), text_item, torch.tensor([self.EOS_IDX])], dim=0))
            mms_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), mms_item, torch.tensor([self.EOS_IDX])], dim=0))
        text_batch = pad_sequence(text_batch, padding_value=self.PAD_IDX)
        mms_batch = pad_sequence(mms_batch, padding_value=self.PAD_IDX)

        return text_batch, mms_batch


    def greedy_decode(self, model, src, src_mask, max_len, start_symbol):
        src = src.to(device)
        src_mask = src_mask.to(device)
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
        for i in range(max_len-1):
            memory = memory.to(device)
            memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                                        .type(torch.bool)).to(device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.item()
            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys


    def translate(self, model, src, src_vocab, tgt_vocab, src_tokenizer, tokenization):
        model.eval()
        tokens = [self.BOS_IDX] + [src_vocab.get_stoi()[tok] for tok in src_tokenizer.encode(src, out_type=str)]+ [self.EOS_IDX]
        num_tokens = len(tokens)
        src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=self.BOS_IDX).flatten()
        tgt_tokens_filtered = [token for token in tgt_tokens if token not in [self.BOS_IDX, self.EOS_IDX]]
        if tokenization == Tokenization.SOURCE_TARGET:
            return " ".join([tgt_vocab.get_itos()[tok] for tok in tgt_tokens_filtered])
        elif tokenization == Tokenization.SOURCE_ONLY:
            return "".join([tgt_vocab.lookup_token(tok) for tok in tgt_tokens_filtered])
        else:
            raise ValueError("Invalid tokenization mode")    


    def train_epoch(self, model, train_iter, optimizer):
        model.train()
        losses = 0
        for idx, (src, tgt) in enumerate(train_iter):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,
                                        src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:,:]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            
            if not self.augment:
                losses += loss    
                losses.backward()

            else:    
                losses += loss.item()                
                loss.backward()

            optimizer.step()

        return losses / len(train_iter)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    def create_transformer(self):
        SRC_VOCAB_SIZE = len(self.text_vocab)
        TGT_VOCAB_SIZE = len(self.mms_vocab)  
        EMB_SIZE = 512
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

        return transformer


NHEAD = 8


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
