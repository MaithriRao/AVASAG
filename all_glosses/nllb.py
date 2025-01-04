import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
import os 
from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu
from . import datasets
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import time
from enum import Enum, verify, UNIQUE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = 'facebook/nllb-200-distilled-600M' #for nllb
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def train(fold, ds,  augment):

    if not augment:
        augment_dir = "original_data"
    else:
        augment_dir = "aug_data"
    save_folder = os.path.join("/ds/videos/AVASAG/allgloss_tg/", augment_dir, "nllb") 
    save_file_path = os.path.join(save_folder, "result")
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    (original, modified, full) = ds
    dataset = original

    # Split the dataset into 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(kf.split(dataset))

    # Split the dataset into train and test sets based on the current fold
    train_indices = [idx for fold_idx, idx in enumerate(folds[fold][0]) if fold_idx != fold]
    test_indices = folds[fold][1]
    train_data = [dataset[idx] for idx in train_indices]
    test_data = [dataset[idx] for idx in test_indices]

    # Augment the training data if augment=True
    if augment:
        train_data = augment_data(train_data, modified)
        
    train_dataset = datasets.SignLanguageDataset(train_data, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=datasets.collate_fn)

    test_dataset = datasets.SignLanguageDataset(test_data, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=datasets.collate_fn)

    NUM_EPOCHS = 1000
    loss_graf = []

    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_log = open(save_file_path+ f"_fold_{fold}_train_log.txt", 'w')

    best_epoch = 0

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, tokenizer)

        end_time = time.time()
        log = "Epoch: " + str(epoch)+", Train loss: "+ str(train_loss)+" Epoch duration "+ str(end_time - start_time)+"\n"
        train_log.write(log)
        if epoch <= 1 or train_loss < min(loss_graf):
            best_model_path = save_file_path+f"_fold_{fold}_best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            log = "min so far is at epoch: "+ str(epoch)+"\n"
            train_log.write(log)
            best_epoch = epoch

        loss_graf.append(train_loss)

    log = "best epoch is: "+ str(best_epoch)
    train_log.write(log)
    train_log.close()

    torch.save(model.state_dict(), save_file_path+f"_fold_{fold}_last_model.pt")

    return test_dataloader, save_file_path
    
def extract_glosses(glosses):
    main_glosses, dom_glosses, ndom_glosses = [], [], []
    for gloss in glosses.split():
        glosses_split = gloss.split("_")
        if len(glosses_split) > 0:
            main_glosses.append(glosses_split[0])
        if len(glosses_split) > 1:
            dom_glosses.append(glosses_split[1])
        if len(glosses_split) > 2:
            ndom_glosses.append(glosses_split[2])
    return " ".join(main_glosses), " ".join(dom_glosses), " ".join(ndom_glosses)



def count_length_comparisons(hypotheses, ground_truths):
    counts = {
        'num_P_T': sum(len(h.split()) > len(g.split()) for h, g in zip(hypotheses, ground_truths)),
        'num_T_P': sum(len(h.split()) < len(g.split()) for h, g in zip(hypotheses, ground_truths)),
        'num_e': sum(len(h.split()) == len(g.split()) for h, g in zip(hypotheses, ground_truths))
    }
    return counts     

def save_results(fold, model_type, save_file_path, counts, bleus, ground_truths, hypotheses):
    with open(save_file_path + f"_fold_{fold}_{model_type}_outputs.txt", "w") as f:
        # Write BLEU scores for each gloss type
        f.write("BLEU Scores:\n")
        for gloss_type, score in bleus.items():
            f.write(f"{gloss_type}: {score}\n")
        
        f.write("\nLength Comparison Counts:\n")
        # Write counts for each gloss type
        for gloss_type, count_dict in counts.items():
            f.write(f"{gloss_type}:\n")
            f.write(f"  P>T: {count_dict['num_P_T']}\n")
            f.write(f"  T>P: {count_dict['num_T_P']}\n")
            f.write(f"  Equal: {count_dict['num_e']}\n")

        f.write("\nGround Truth and Predicted Texts:\n")
        # Write ground truth and predictions for each sample
        for i in range(len(ground_truths['maingloss'])):
            f.write(f"\nSample {i+1}:\n")
            f.write(f"Ground Truth (maingloss): {ground_truths['maingloss'][i]}\n")
            f.write(f"Predicted (maingloss): {hypotheses['maingloss'][i]}\n")
            f.write(f"Ground Truth (domgloss): {ground_truths['domgloss'][i]}\n")
            f.write(f"Predicted (domgloss): {hypotheses['domgloss'][i]}\n")
            f.write(f"Ground Truth (ndomgloss): {ground_truths['ndomgloss'][i]}\n")
            f.write(f"Predicted (ndomgloss): {hypotheses['ndomgloss'][i]}\n")

def calculate_bleu(hypotheses, references):
    scores = []
    for hyp, ref in zip(hypotheses, references):
        ref = [ref.split()]  
        hyp = hyp.split()  
        score = sentence_bleu(ref, hyp, weights=(1, 0, 0, 0))  # BLEU-1
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0  # Average BLEU-1


def evaluate(fold, model_type, model_name, test_dataloader, save_file_path): # Evaluation
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    model.load_state_dict(torch.load(save_file_path+f"_fold_{fold}_{model_type}_{model_name}"))

    ground_truths = {
        'maingloss': [],
        'domgloss': [],
        'ndomgloss': []
    }
    hypotheses = {
        'maingloss': [],
        'domgloss': [],
        'ndomgloss': []
    }

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            file_Id, text_tokens_padded, maingloss_tokens_padded = batch
            text_tokens_padded = text_tokens_padded.to(device)
            maingloss_tokens_padded = maingloss_tokens_padded.to(device)

            pred = model.generate(input_ids=text_tokens_padded, max_length=maingloss_tokens_padded.size(1))

            for i in range(text_tokens_padded.size(0)):
                gt_maingloss = "".join(tokenizer.decode(maingloss_tokens_padded[i], skip_special_tokens=True))
                input_text = tokenizer.decode(text_tokens_padded[i], skip_special_tokens=True)
                text_predicted = tokenizer.decode(pred[i], skip_special_tokens=True)


                main_glosses, dom_glosses, ndom_glosses = extract_glosses(gt_maingloss)
                main_glosses_pred, dom_glosses_pred, ndom_glosses_pred = extract_glosses(text_predicted)


                if fold == 9: # only for printing
                    print("file_Id", file_Id)
                    print(f"\nSample {len(gt_maingloss) + 1}:")
                    print(f"Input Text: {input_text}")

                    print(f"ground_truth_maingloss: {main_glosses}")
                    print(f"ground_truth_domgloss: {dom_glosses}")
                    print(f"ground_truth_ndomgloss: {ndom_glosses}")

                    print(f"main_glosses_pred: {main_glosses_pred}")
                    print(f"dom_glosses_pred: {dom_glosses_pred}")
                    print(f"ndom_glosses_pred: {ndom_glosses_pred}")

                ground_truths['maingloss'].append(main_glosses)
                ground_truths['domgloss'].append(dom_glosses)
                ground_truths['ndomgloss'].append(ndom_glosses)

                hypotheses['maingloss'].append(main_glosses_pred)
                hypotheses['domgloss'].append(dom_glosses_pred)
                hypotheses['ndomgloss'].append(ndom_glosses_pred)                    
 
                
    # Calculate BLEU score 
    bleu = BLEU()
    bleus = {
        'maingloss': bleu.corpus_score(hypotheses['maingloss'], [ground_truths['maingloss']]),
        'domgloss': calculate_bleu(hypotheses['domgloss'], ground_truths['domgloss']),
        'ndomgloss': calculate_bleu(hypotheses['ndomgloss'], ground_truths['ndomgloss'])
    }


    # Count lengths for each gloss type
    counts = {key: count_length_comparisons(hypotheses[key], ground_truths[key]) for key in hypotheses}

    # Save results to file
    save_results(fold, model_type, save_file_path, counts, bleus, ground_truths, hypotheses)

    return bleus['maingloss'].score, bleus['domgloss'], bleus['ndomgloss']


def augment_data(train_data, sentences):

    augmented_train_data = train_data.copy()
    augmented_train_data.extend(sentences)

    return augmented_train_data  

def train_epoch(model, train_dataloader, optimizer, tokenizer):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        file_Id, text_tokens_padded, maingloss_tokens_padded = batch
        text_tokens_padded = text_tokens_padded.to(device)
        maingloss_tokens_padded = maingloss_tokens_padded.to(device)  
        input_attention_mask = (text_tokens_padded != tokenizer.pad_token_id).to(device)

        optimizer.zero_grad()

        output_final  = model(input_ids=text_tokens_padded, attention_mask=input_attention_mask, labels=maingloss_tokens_padded)
        loss = output_final.loss    
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_loss / len(train_dataloader)     
    return avg_train_loss

if __name__ == "__main__":
    import sys

    original_scores = { 'best': { 'maingloss': [], 'domgloss': [], 'ndomgloss': [] },
                    'last': { 'maingloss': [], 'domgloss': [], 'ndomgloss': [] } }

    
    augmented_scores = { 'best': { 'maingloss': [], 'domgloss': [], 'ndomgloss': [] },
                    'last': { 'maingloss': [], 'domgloss': [], 'ndomgloss': [] } }

    ds = datasets.read()    

    for fold in range(10):
        print(f"Current fold {fold}:")
        print("Original data :")
        test_dataloader, save_file_path = train(fold, ds, augment=False)
        test_dataloader_1, save_file_path_1 = train(fold, ds, augment=True)
        assert save_file_path != save_file_path_1
        for model_type in ['best', 'last']:
            print(f"{model_type.capitalize()} model:")
            original_maingloss, original_domgloss,  original_ndomgloss = evaluate(fold, model_type, "model.pt", test_dataloader, save_file_path)
            original_scores[model_type]['maingloss'].append(original_maingloss)
            original_scores[model_type]['domgloss'].append(original_domgloss)
            original_scores[model_type]['ndomgloss'].append(original_ndomgloss)

            aug_maingloss, aug_domgloss,  aug_ndomgloss = evaluate(fold, model_type, "model.pt", test_dataloader_1, save_file_path_1)
            augmented_scores[model_type]['maingloss'].append(aug_maingloss)
            augmented_scores[model_type]['domgloss'].append(aug_domgloss)
            augmented_scores[model_type]['ndomgloss'].append(aug_ndomgloss)


    avg_original_scores = { model_type: { gloss: np.mean(original_scores[model_type][gloss]) for gloss in original_scores[model_type] } for model_type in original_scores }
    avg_augmented_scores = { model_type: { gloss: np.mean(augmented_scores[model_type][gloss]) for gloss in augmented_scores[model_type] } for model_type in augmented_scores }


    for model_type in ['best', 'last']:
        for gloss in ['maingloss', 'domgloss', 'ndomgloss']:
            print(f" BLEU score on original data for each fold {model_type}_model {gloss}: {original_scores[model_type][gloss]}")
            print(f" BLEU score on augmented data for each fold {model_type}_model {gloss}: {augmented_scores[model_type][gloss]}")
            print(f" Average BLEU score on original data for {model_type}_model {gloss}: {avg_original_scores[model_type][gloss]}")
            print(f" Average BLEU score on augmented data for {model_type}_model {gloss}: {avg_augmented_scores[model_type][gloss]}")
         