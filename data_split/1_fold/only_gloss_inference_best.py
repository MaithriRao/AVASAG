
import math
import pickle
import torchtext
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from torch import Tensor
import io
import time
import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from typing import List
from sacrebleu.metrics import BLEU
import numpy as np
from .. import datasets
from ..model import Model
from ..utils import Tokenization
from sklearn.model_selection import train_test_split



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128

def best_model(ds, tokenization, augment):

    if tokenization == Tokenization.SOURCE_ONLY:
        tokenization_dir = "source_only"
    elif tokenization == Tokenization.SOURCE_TARGET:
        tokenization_dir = "source_target"
    else:
        raise ValueError("Invalid tokenization value")

    model = Model(ds, augment)

    (original, modified, full) = ds
    (tokenizer_original, vocab_original, sentences_original) = original
    (tokenizer_full, vocab_full, sentences_full) =  full
    (source_text_full, target_gloss_full) = sentences_full
    (source_text_original, target_gloss_original) = sentences_original
    

    if augment:
        source_train, source_test, target_train, target_test = train_test_split(source_text_full, target_gloss_full, test_size=0.25, random_state = 42)
    else:
        source_train, source_test, target_train, target_test = train_test_split(source_text_original, target_gloss_original, test_size=0.25, random_state = 42)
  
    transformer = model.create_transformer()
    transformer = transformer.to(device)

    augment_or_original_dir = "aug_data" if augment else "original_data"

    save_folder_path = os.path.join("data_split", "1_fold", tokenization_dir, augment_or_original_dir, "onlyGloss")
    model_file_path = os.path.join(save_folder_path, "result_best_model.pt")

    transformer.load_state_dict(torch.load(model_file_path))

    ground_truth = []
    hypothesis = []
    preds_file = open(save_folder_path+"_predictions.txt", "w")

    num_P_T = 0
    num_T_P = 0
    num_e = 0

    for de_text, gl_text in zip(source_test, target_test):
        if tokenization == Tokenization.SOURCE_TARGET:
            source_entry = de_text[1]
            target_entry = gl_text[1]

            print(f"Source Sententence : {source_entry}")
            print(f"Target Sententence : {target_entry}")

            gl_pred = model.translate(transformer, source_entry, model.text_vocab, model.mms_vocab, model.text_tokenizer, tokenization)
            print(f"gloss prediction   : {gl_pred}")

            translated_sentence = ""
            for char in gl_pred:
                if char == "▁":
                    translated_sentence += " "
                elif char != " ":
                    translated_sentence += char

            translated_sentence = translated_sentence.strip()

            
            print(f"translated_sentence: {translated_sentence}")

            ground_truth.append(target_entry)
            hypothesis.append(translated_sentence)

            P = len(translated_sentence.split())
            T = len(target_entry.split())

        elif tokenization == Tokenization.SOURCE_ONLY:
            source_entry = de_text[1]
            target_entry = "".join(gl_text[1])

            print(f"Source Sententence: {source_entry}")
            print(f"Target Sententence: {target_entry}")

            gl_pred = model.translate(transformer, source_entry, model.text_vocab, model.mms_vocab, model.text_tokenizer, tokenization)
            print(f"Predicted gloss   : {gl_pred}")

            ground_truth.append(target_entry)
            hypothesis.append(gl_pred)

            P = len(gl_pred.split())
            T = len(target_entry.split())

        else:
            raise ValueError("Invalid tokenization value")
  
        if P > T:
            print("P:", P)
            num_P_T += 1
        elif T > P:
            print("T:", T)
            num_T_P += 1
        else:
            num_e += 1

            
        preds_file.write(str(de_text[0])+"\n")
        preds_file.write(de_text[1]+"\n")
        preds_file.write(target_entry+"\n")
        preds_file.write(gl_pred+"\n")
        preds_file.write("************************************\n")
    preds_file.close()



    f = open(save_folder_path+"_outputs.txt","w")

    line = "P>T: "+ str(num_P_T) +"\n"
    f.write(line)

    line = "T>P: "+ str(num_T_P) +"\n"
    f.write(line)

    line = "equal: "+ str(num_e) +"\n"
    f.write(line)

    from sacrebleu.metrics import BLEU

    # use the lists ground_truth, hypothesis
    refs = [ground_truth]

    bleu = BLEU()

    result = bleu.corpus_score(hypothesis, refs)

    line = "BLEU score for maingloss: "+str(result)+"\n"
    f.write(line)

    f.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python only_gloss_inference_best.py [--source-only|--source-target]")
        sys.exit(1)

    if sys.argv[1] == "--source-only":
        print("Using source only")
        tokenization = Tokenization.SOURCE_ONLY
    elif sys.argv[1] == "--source-target":
        print("Using source and target")
        tokenization = Tokenization.SOURCE_TARGET
    else:
        print("You have to specify either --source-only or --source-target as an argument.")
        sys.exit(1)
        
    ds = datasets.read(tokenization)
    print("Original data :")
    best_model(ds, tokenization, augment=False)

    print("Augmented data:")
    best_model(ds, tokenization, augment=True)
