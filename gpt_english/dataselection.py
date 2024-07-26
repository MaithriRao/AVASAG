import math
import torch
from torch import Tensor
import io
import time
import os
import pandas as pd
import json
from datetime import datetime
import pickle
from pathlib import Path
from torch.utils.data import Dataset
# from transformers import BertTokenizer
from collections import Counter
from itertools import chain
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from transformers import GPT2Tokenizer
import torch.nn.functional as F

features_names = ["maingloss", "domgloss", "ndomgloss", "domreloc","ndomreloc", "framestart", "headmov", "domhandrelocx", "domhandrelocy", "domhandrelocz", "domhandrelocax", 
                  "domhandrelocay", "domhandrelocaz", "domhandrelocsx", "domhandrelocsy", "domhandrelocsz",
                  "domhandrotx", "domhandroty", "domhandrotz", 
                  "ndomhandrelocx", "ndomhandrelocy", "ndomhandrelocz", "ndomhandrelocax", 
                  "ndomhandrelocay", "ndomhandrelocaz", "ndomhandrelocsx", "ndomhandrelocsy", "ndomhandrelocsz",
                  "ndomhandrotx", "ndomhandroty", "ndomhandrotz"]

HEADMOV_TO_INT = {'0': 0, '1': 1, '2': 2, 'no': 3, 'yes': 4} 

directory = "mms-subset91"
text_directory = "annotations_full/annotations"


def load_data(directory, text_directory, features_names):
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
    return data_list


def normalize_inflection_features(data_x, features_names):
    
    inflection_stats = {}
    for feature in features_names[7:]:
        all_values = []
        for data in data_x:
            if feature in data and data[feature]:
                all_values.extend([v for v in data[feature] if not np.isnan(v)])
        if all_values:
            global_min = min(all_values)
            global_max = max(all_values)
            mean_value = np.mean(global_min + global_max)
            inflection_stats[feature] = {
                'global_min': global_min,
                'global_max': global_max,
                'mean_value': mean_value
            }
            print(f"{feature} - Global min: {global_min:.4f}, Global max: {global_max:.4f}, Mean: {mean_value:.4f}")
        else:
            print(f"Warning: No valid values found for {feature}")
            inflection_stats[feature] = None

    # Preprocess the data to handle missing values
    for data in data_x: 
        for feature in features_names[7:]:
            if inflection_stats[feature] is None:
                continue

            if feature not in data or not data[feature]:
                # If missing values, replace with mean value of text length
                data[feature] = [inflection_stats[feature]['mean_value']] * len(data['text'].split())
            else:
                normalized_values = []
                for j, value in enumerate(data[feature]):
                    if np.isnan(value):
                        normalized_values.append(inflection_stats[feature]['mean_value'])
                    else:
                        normalized_value = (value - inflection_stats[feature]['global_min']) / (inflection_stats[feature]['global_max'] - inflection_stats[feature]['global_min'])
                        if np.isnan(normalized_value):
                            print(f"Normalization resulted in NaN in file: {data.get('file_ID', 'Unknown')} for feature {feature} at index {j}. Value: {value}")
                            normalized_value = inflection_stats[feature]['mean_value']
                        normalized_values.append(normalized_value)
                data[feature] = normalized_values

    return data_x, inflection_stats

# Tokenize text data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
data_list = load_data(directory, text_directory, features_names)
# Split data into training and validation sets
train_data, val_data = train_test_split(data_list, test_size=0.25, random_state=42)
normalized_train_data, inflection_stats = normalize_inflection_features(train_data, features_names)


# Define custom dataset class
class SignLanguageDataset(Dataset):
    def __init__(self, data_list, tokenizer, inflection_stats, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(tokenizer)
        self.framestart_scaler = StandardScaler()        
        all_framestart = [item for data in data_list for item in data['framestart']]
        self.framestart_scaler.fit(np.array(all_framestart).reshape(-1, 1))
        self.inflection_stats = inflection_stats     
 
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        file_Id = data['file_ID']
        text_tokens = self.tokenizer.encode(data['text'], add_special_tokens=True, max_length=self.max_length, truncation=True)
        text_tokens = torch.tensor(text_tokens)

        gloss_feature = ["maingloss", "domgloss", "ndomgloss"]
        gloss_tokens = {}
        for feature in gloss_feature:
   
            if feature in data:
                tokens = self.tokenizer.encode(' '.join(data[feature]), add_special_tokens=True, max_length=self.max_length, truncation=True)
                gloss_tokens[feature] = torch.tensor(tokens)

        feature_tensors = {}
        for feature in self.inflection_stats.keys():
            if feature in data:
                feature_tensors[feature] = torch.tensor(data[feature], dtype=torch.float32)

        framestart = self.framestart_scaler.transform(np.array(data['framestart']).reshape(-1, 1)).flatten()
        framestart = torch.tensor(framestart, dtype=torch.float32)

        boolean_features = {}
        for feature in ['domreloc','ndomreloc']:
            if feature in data:
                boolean_features[feature]= torch.tensor([transform_bool_feature(val) for val in data[feature]], dtype=torch.long)

        return file_Id, text_tokens, gloss_tokens, framestart, boolean_features, feature_tensors

    def get_scaler(self):
        return self.framestart_scaler 

    def get_inflection_stats(self):
        return self.inflection_stats


def transform_bool_feature(val):
    val = val.lower().strip()
    return HEADMOV_TO_INT.get(val, 0)        


def collate_fn(batch):
    file_Id, text_tokens, gloss_tokens, framestart, boolean_features, feature_tensors= zip(*batch)

    padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=padding_value)

    gloss_tokens_padded = {}
    for feature in ['maingloss', 'domgloss', 'ndomgloss']:
        if all(feature in sample for sample in gloss_tokens):
            gloss_tokens_padded[feature] = pad_sequence([sample[feature] for sample in gloss_tokens], 
                                                        batch_first=True, 
                                                        padding_value=padding_value)
    # Ensure all have the same sequence length
    max_len = max(text_tokens_padded.size(1), max(tensor.size(1) for tensor in gloss_tokens_padded.values()))

    text_tokens_padded = torch.nn.functional.pad(text_tokens_padded, (0, max_len - text_tokens_padded.size(1)), value=padding_value)
    
    for feature in gloss_tokens_padded:
        gloss_tokens_padded[feature] = torch.nn.functional.pad(gloss_tokens_padded[feature], (0, max_len - gloss_tokens_padded[feature].size(1)), value=padding_value)
    
    boolean_features_padded = {}
    for feature in boolean_features[0].keys():
        feature_list = [sample[feature] for sample in boolean_features]
        padded_feature = pad_sequence(feature_list, batch_first=True, padding_value=-1)
        padded_feature = torch.nn.functional.pad(padded_feature, (0, max_len - padded_feature.size(1)), value=-1)
        boolean_features_padded[feature] = padded_feature

    framestart_padded = pad_sequence(framestart, batch_first=True, padding_value=0.0)
    framestart_padded = torch.nn.functional.pad(framestart_padded, (0, max_len - framestart_padded.size(1)), value=0.0)


    inflection_features_padded = {}
    for feature in feature_tensors[0].keys():
        feature_list = [sample[feature] for sample in feature_tensors]
        padded_feature = pad_sequence(feature_list, batch_first=True, padding_value=0.0)
        padded_feature = torch.nn.functional.pad(padded_feature, (0, max_len - padded_feature.size(1)), value=0.0)
        inflection_features_padded[feature] = padded_feature
 
    return file_Id, text_tokens_padded, gloss_tokens_padded, framestart_padded, boolean_features_padded, inflection_features_padded


# Create DataLoader instances
train_dataset = SignLanguageDataset(train_data, tokenizer, inflection_stats)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

val_dataset = SignLanguageDataset(val_data, tokenizer, inflection_stats)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
       