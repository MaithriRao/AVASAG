#!/usr/bin/env python
# coding: utf-8

# In[29]:


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

torch.manual_seed(0)
# PyTorch operations must use “deterministic” algorithms. if not available throw RuntimeError
# torch.use_deterministic_algorithms(True)


from datetime import datetime

save_folder_address = "inference"+str(datetime.now()).replace(" ", "__")


# In[30]:


print("haha")


# In[31]:


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


# data_list is a list of dictionaries\
# each dictianry corresponds to a data sample in the dataset\
# file_ID is the file number, text is the german sentence, and the rest are all a list of the same length containing different values of gloss, boolean, and real value numbers.

# In[32]:


boolean_map = {"yes": 1, "no": 0}
for data in data_list:
    data["domreloc"] = [boolean_map[value] for value in data["domreloc"]]
    data["ndomreloc"] = [boolean_map[value] for value in data["ndomreloc"]]
#     data["shoulders"] = [boolean_map[value] for value in data["shoulders"]]


# In[33]:


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


# In[34]:


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


# In[35]:


# get spacy tokenizer for German text
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

de_vocab = build_German_vocab(data_list, de_tokenizer)
gl_vocab = build_gloss_vocab(data_list, "maingloss")
dom_vocab = build_gloss_vocab(data_list, "domgloss")
ndom_vocab = build_gloss_vocab(data_list, "ndomgloss")


# I added this two lines because some tokens from the validation and test are not in the train set.

# In[36]:


UNK_IDX = de_vocab['<unk>']
de_vocab.set_default_index(UNK_IDX)
gl_vocab.set_default_index(UNK_IDX)
dom_vocab.set_default_index(UNK_IDX)
ndom_vocab.set_default_index(UNK_IDX)


# In[37]:


# divide train and test here
import math
import random

div = math.floor(len(data_list)*0.75)

data_list_copy = data_list.copy()

random.seed(1)
random.shuffle(data_list_copy)

test_data_raw = data_list_copy[div:len(data_list)]
train_data_raw = data_list_copy[0:div]


# In[38]:


import json

with open("test_data.json", "w") as outfile:
    json.dump(test_data_raw, outfile)
    
with open("train_data.json", "w") as outfile:
    json.dump(train_data_raw, outfile)


# In[39]:


with open('test_data.json', 'r') as openfile:
    test_data_raw = json.load(openfile)
    
print(len(test_data_raw))


with open('train_data.json', 'r') as openfile:
    train_data_raw = json.load(openfile)
    
print(len(train_data_raw))


# In[8]:


# with open('../Downloads/test_data.json', 'r') as openfile:
#     json_object_server = json.load(openfile)
    
# print(len(json_object_server))


# In[10]:


# for item in json_object_server:
#     print(item['file_ID'])
#     print(item["text"])
#     print(item["maingloss"])


# In[ ]:




