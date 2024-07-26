import torch
import torch.nn as nn
from .model_series import *
from . import dataselection
import numpy as np
from transformers import GPT2Tokenizer


inflection_features = ["domhandrelocx", "domhandrelocy", "domhandrelocz", "domhandrelocax", 
                  "domhandrelocay", "domhandrelocaz", "domhandrelocsx", "domhandrelocsy", "domhandrelocsz",
                  "domhandrotx", "domhandroty", "domhandrotz", 
                  "ndomhandrelocx", "ndomhandrelocy", "ndomhandrelocz", "ndomhandrelocax", 
                  "ndomhandrelocay", "ndomhandrelocaz", "ndomhandrelocsx", "ndomhandrelocsy", "ndomhandrelocsz",
                  "ndomhandrotx", "ndomhandroty", "ndomhandrotz"]
gloss_feature = ["maingloss", "domgloss", "ndomgloss"] 

boolean_features = ["domreloc","ndomreloc"]

class Configs:
    def __init__(self):
        self.d_model = 768  
        self.lstm_hidden_size = 512 
        self.lstm_layers = 4
        self.transformer_layers = 4  
        self.transformer_heads = 6
        self.vocab_size = 10000  
        self.dropout = 0.1
        self.is_gpt = True
        self.pretrain = True
        self.gpt_layers = 4
        self.freeze = True 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
configs = Configs()
configs.vocab_size = tokenizer.vocab_size  
model = GPT2TS(configs, device, gloss_feature, boolean_features, inflection_features)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,  betas=(0.9, 0.98), eps=1e-03)
print_every = 5
best_train_loss = float('inf')
best_model_path = 'best_model_gpt.pth'
final_model_path = 'final_model_gpt.pth'

# Loss functions
criterion_gloss = nn.CrossEntropyLoss(ignore_index=0)
criterion_boolean_feature = nn.CrossEntropyLoss(ignore_index=-1)

def domhandrelocx_criterion(pred, target, padding_value=0):
    mask = (target != padding_value).float()
    loss = F.mse_loss(pred, target, reduction='none')
    masked_loss = (loss * mask).sum() / mask.sum().clamp(min=1e-8)
    return masked_loss

def compute_boolean_feature_loss(pred, target):
    loss = criterion_boolean_feature(pred.view(-1, pred.size(-1)), target.view(-1))
    return loss    

def feature_loss(pred_values, target):
    min_len = min(pred_values.size(1), target.size(1))
    pred_values = pred_values[:, :min_len]
    target = target[:, :min_len]
    
    mask = (target != 0).float()
    loss = F.mse_loss(pred_values * mask, target, reduction='none')
    return loss.sum() / (mask.sum() + 1e-8)

def compute_loss(predictions, targets):
    framestart_pred  = predictions
    framestart_target  = targets
    loss_framestart = feature_loss(framestart_pred, framestart_target)
    return loss_framestart
  

for epoch in range(1000):
    model.train()
    for batch_idx, batch in enumerate(dataselection.train_dataloader):
        file_Id, text_tokens_padded, gloss_tokens_padded, framestart_padded, boolean_features_padded, inflection_feature_padded = batch
        text_tokens_padded = text_tokens_padded.to(device)
        gloss_tokens_padded = {k: v.to(device) for k, v in gloss_tokens_padded.items()}
        framestart_padded = framestart_padded.to(device)
        boolean_features_padded = {k: v.to(device) for k, v in boolean_features_padded.items()}
        inflection_feature_padded = {feature: inflection_feature_padded[feature].to(device) for feature in inflection_features}

        optimizer.zero_grad()
        gloss_logits, framestart_pred, boolean_pred, inflection_feature_preds = model(text_tokens_padded)

        gloss_losses = {}
        for gloss, logits in gloss_logits.items():
            gloss_losses[gloss] = criterion_gloss(logits.view(-1, logits.size(-1)), gloss_tokens_padded[gloss].view(-1))
        total_gloss_loss = sum(gloss_losses.values())

        boolean_losses = {}
        for boolean_feature, pred in boolean_pred.items():
            boolean_losses[boolean_feature] = compute_boolean_feature_loss(pred.view(-1, pred.size(-1)), boolean_features_padded[boolean_feature].view(-1))
        total_boolean_loss = sum(boolean_losses.values())   

        loss_framestart = compute_loss(predictions=(framestart_pred),
            targets=(framestart_padded))

        inflection_feature_losses = {}
        for feature, pred in inflection_feature_preds.items():
            inflection_feature_losses[feature] = domhandrelocx_criterion(pred, inflection_feature_padded[feature])
        total_inflection_loss = sum(inflection_feature_losses.values())    
  
        total_loss = total_gloss_loss + loss_framestart + total_boolean_loss + total_inflection_loss

        total_loss.backward()
        optimizer.step()
          
    avg_train_loss = total_loss / len(dataselection.train_dataloader)
    print(f"Training Loss at {epoch + 1}: {avg_train_loss:.4f}")
    if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_epoch} with training loss: {best_train_loss:.4f}")

torch.save(model.state_dict(), final_model_path)
print("Final model saved after for last epoch")
        