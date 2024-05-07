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
import sentencepiece as spm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pickle
from pathlib import Path


features_names = ["maingloss", "domgloss", "ndomgloss", "domreloc", "ndomreloc",
                  "domhandrelocx", "domhandrelocy", "domhandrelocz", "domhandrelocax", 
                  "domhandrelocay", "domhandrelocaz", "domhandrelocsx", "domhandrelocsy", "domhandrelocsz",
                  "domhandrotx", "domhandroty", "domhandrotz", 
                  "ndomhandrelocx", "ndomhandrelocy", "ndomhandrelocz", "ndomhandrelocax", 
                  "ndomhandrelocay", "ndomhandrelocaz", "ndomhandrelocsx", "ndomhandrelocsy", "ndomhandrelocsz",
                  "ndomhandrotx", "ndomhandroty", "ndomhandrotz"]


def read(text_info, mms_info):
    data_list = []
    (text_directory, text_encoding) = text_info
    (mms_directory, mms_encoding) = mms_info
    for filenumber in os.listdir(text_directory):
        f = os.path.join(mms_directory, filenumber+".mms")
        try:
            df = pd.read_csv(f, encoding=mms_encoding)
        except FileNotFoundError as e:
            print(f"WARNING: Text file exists while mms file does not, skipping: {e}")
            continue   

        text_address = os.path.join(text_directory, filenumber, "gebaerdler.Text_Deutsch.annotation~")
        file = open(text_address, encoding=text_encoding)
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
    return data_list


def train_tokenizers(data_list):
    text_list = []
    gloss_list = []
    for data_dict in data_list:
        german_text = data_dict["text"]
        text_list.append(german_text)

        glosses = data_dict["maingloss"]
        gloss_text = " ".join(glosses)
        gloss_list.append(gloss_text)

    text_model = io.BytesIO()
    gloss_model = io.BytesIO()

    spm.SentencePieceTrainer.train(sentence_iterator=iter(text_list), model_writer=text_model, vocab_size=100)
    spm.SentencePieceTrainer.train(sentence_iterator=iter(gloss_list), model_writer=gloss_model, vocab_size=100)

    text_tokenizer = spm.SentencePieceProcessor(model_proto=text_model.getvalue())
    mms_tokenizer = spm.SentencePieceProcessor(model_proto=gloss_model.getvalue())

    vocab_size = text_tokenizer.get_piece_size()
    print(f"Vocabulary size for the text used now is: {vocab_size}")

    vocab_size_mms = mms_tokenizer.get_piece_size()
    print(f"Vocabulary size for the mms used now is: {vocab_size_mms}")

    return (text_tokenizer, mms_tokenizer)

def split_dataset(data_list):
    source_text = []
    target_gloss = []
    for data in data_list:
        text_data = (data["file_ID"], data["text"])
        source_text.append(text_data)

        maingloss_str = ' '.join(data["maingloss"])
        maingloss_data = (data["file_ID"], maingloss_str)
        target_gloss.append(maingloss_data)

    assert len(source_text) == len(target_gloss), "Number of source and target sentences do not match!"
    return source_text, target_gloss


def build_text_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        tokenized_text = sentence[1]
        counter.update(tokenizer.encode(tokenized_text, out_type=str))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

#tokenization to both german text and gloss
def build_mms_vocab_with_tokenizer(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer.encode(sentence[1], out_type=str)
        counter.update(tokens)
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])  

#tokenization only to german text
def build_mms_vocab_without_tokenizer(sentences):
    counter = Counter()
    for sentence in sentences:
        tokens = sentence[1] 
        counter.update(tokens)
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])      
    