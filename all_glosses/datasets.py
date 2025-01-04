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
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import numpy as np
from transformers import AutoTokenizer
import torch.nn.functional as F
from pathlib import Path
from . import data_selection

mms_directories = [
    ("mms-subset91", 'latin-1'),
    ("modified/location/mms", 'utf-8'),
    ("modified/platform/mms", 'utf-8'),
    ("modified/time/mms", 'utf-8'),
    ("modified/train_name/mms", 'utf-8'),
]
text_directories = [
    ("annotations_full/annotations", 'latin-1'),
    ("modified/location/text", 'utf-8'),
    ("modified/platform/text", 'utf-8'),
    ("modified/time/text", 'utf-8'),
    ("modified/train_name/text", 'utf-8'),
]

checkpoint = 'facebook/nllb-200-distilled-600M' #for nllb
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def read():
    data_list_only_original = []
    data_list_only_modified = []
    for i, text_info in enumerate(text_directories):
        mms_info = mms_directories[i]
        data_list_one = data_selection.read(text_info, mms_info)
        if i <= 0:
            data_list_only_original += data_list_one
        else:
            data_list_only_modified += data_list_one

    data_list_full = data_list_only_original + data_list_only_modified

    return (data_list_only_original, data_list_only_modified, data_list_full)


class SignLanguageDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(tokenizer)
 
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        file_Id = data['file_ID']
        text_tokens = self.tokenizer.encode(data['text'], add_special_tokens=True)
        text_tokens = torch.tensor(text_tokens)

        maingloss_tokens = self.tokenizer.encode(''.join(data['gloss']).lower(), add_special_tokens=True)
        maingloss_tokens = torch.tensor(maingloss_tokens)

        return file_Id, text_tokens, maingloss_tokens

        return file_Id, text_tokens, gloss_tokens


def collate_fn(batch):
    file_Id, text_tokens, gloss_tokens = zip(*batch)
    padding_value = tokenizer.pad_token_id  # here for nllb paddign token is 1

    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=padding_value)
    gloss_tokens_padded = torch.nn.utils.rnn.pad_sequence(gloss_tokens, batch_first=True, padding_value=padding_value)

    # Ensure all have the same sequence length
    max_len = max(text_tokens_padded.size(1), gloss_tokens_padded.size(1))

    text_tokens_padded = torch.nn.functional.pad(text_tokens_padded, (0, max_len - text_tokens_padded.size(1)), value=padding_value)
    gloss_tokens_padded = torch.nn.functional.pad(gloss_tokens_padded, (0, max_len - gloss_tokens_padded.size(1)), value=padding_value)

    return file_Id, text_tokens_padded, gloss_tokens_padded
