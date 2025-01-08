import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
import os
import pandas as pd
import json
from datetime import datetime
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .utils import Translation

features_names = ["maingloss"]
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

def read(text_info, mms_info, translation):
    data_list = []
    (text_directory, text_encoding) = text_info
    print("text_directory: ", text_directory)
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
        for feature in features_names:
            gloss_line = " ".join(df["maingloss"].tolist())
        if translation == Translation.TextToGloss:
            combined_line = f"{text_line} ###> {gloss_line}"  # text to gloss
        elif translation == Translation.GlossToText:
            combined_line = f"{gloss_line} ###> {text_line}"  # gloss to text
        else:
            raise ValueError("Invalid translation")
        data_list.append({"text": combined_line})
    return data_list

def create_datasets(translation):
    data_list_only_original = []
    data_list_only_modified = []
    for i, text_info in enumerate(text_directories):
        mms_info = mms_directories[i]
        data_list_one = read(text_info, mms_info, translation)
        if i <= 0:
            data_list_only_original += data_list_one
        else:
            data_list_only_modified += data_list_one

    data_list_full = data_list_only_original + data_list_only_modified


    train_data, temp_data = train_test_split(data_list_full, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=1/3, random_state=42)


    if translation == Translation.TextToGloss:
        translation_dir = "t2g_llama"
    elif translation == Translation.GlossToText:
        translation_dir = "g2t_llama"
    else:
        raise ValueError("Invalid translation")
    with open(f"train_data_{translation_dir}.json", "w") as f:
        json.dump(train_data, f)

    with open(f"val_data_{translation_dir}.json", "w") as f:
        json.dump(val_data, f)

    with open(f"test_data_{translation_dir}.json", "w") as f:
        json.dump(test_data, f)
