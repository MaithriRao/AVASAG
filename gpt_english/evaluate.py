
import torch
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from .model_series import *
from . import dataselection
from transformers import GPT2Tokenizer
import numpy as np
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

features_names = ["maingloss", "domgloss", "ndomgloss", "domreloc","ndomreloc", "framestart", "headmov", "domhandrelocx", "domhandrelocy", "domhandrelocz", "domhandrelocax", 
                  "domhandrelocay", "domhandrelocaz", "domhandrelocsx", "domhandrelocsy", "domhandrelocsz",
                  "domhandrotx", "domhandroty", "domhandrotz", 
                  "ndomhandrelocx", "ndomhandrelocy", "ndomhandrelocz", "ndomhandrelocax", 
                  "ndomhandrelocay", "ndomhandrelocaz", "ndomhandrelocsx", "ndomhandrelocsy", "ndomhandrelocsz",
                  "ndomhandrotx", "ndomhandroty", "ndomhandrotz"]
inflection_features = [feature for feature in features_names if feature not in ["maingloss", "domgloss", "ndomgloss", "domreloc","ndomreloc", "framestart","headmov"]] 
gloss_feature = ["maingloss", "domgloss", "ndomgloss"]    
boolean_features = ["domreloc","ndomreloc"]
INT_TO_HEADMOV = {0: '0', 1: '1', 2: '2', 3: 'no', 4: 'yes'}

class Configs:
    def __init__(self):
        self.d_model = 768  
        self.lstm_hidden_size = 512 
        self.lstm_layers = 4
        self.transformer_layers = 4  
        self.transformer_heads = 6 #from 8 
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
best_model_path = 'best_model_gpt.pth'
model.load_state_dict(torch.load(best_model_path))

ground_truth = []
hypothesis = []
num_P_T = 0  # predicted length > true length
num_T_P = 0  # true length > predicted length
num_e = 0    # equal lengths
framestart_scaler = dataselection.val_dataset.get_scaler()
normalized_data, inflection_stats = dataselection.normalize_inflection_features(dataselection.val_data, features_names)


def denormalize_and_remove_padding(normalized_values, feature_stats):
    global_min = feature_stats['global_min']
    global_max = feature_stats['global_max']    
    # Denormalize
    denormalized = normalized_values * (global_max - global_min) + global_min
    # Remove padding 
    non_zero_mask = normalized_values != 0
    return denormalized[non_zero_mask]


def map_headmov_to_original(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):

        return [INT_TO_HEADMOV[min(max(int(v), 0), 4)] for v in value.flatten()]
    else:
    
        return INT_TO_HEADMOV[min(max(int(value), 0), 4)]

#denrmalization for framestart
def denormalize_feature(normalized_values, scaler):
    # Create a mask for non-zero values
    non_zero_mask = normalized_values != 0
    non_zero_values = normalized_values[non_zero_mask].reshape(-1, 1)
    
    denormalized_non_zero = scaler.inverse_transform(non_zero_values).flatten()
    
    result = np.zeros_like(normalized_values)
    
    result[non_zero_mask] = denormalized_non_zero
    
    return result  

def calculate_accuracy(predictions, targets, ignore_index=-1):
    mask = targets != ignore_index
    correct = (predictions[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct, total

mse_framestart_values = []
mse_framestart_values_denorm = []
print("\nEvaluating on validation data:")
model.eval()
mse_losses = {feature: [] for feature in inflection_stats.keys()}
mse_losses_denorm = {feature: [] for feature in inflection_stats.keys()}

ground_truth = {
    'maingloss': [],
    'domgloss': [],
    'ndomgloss': []
}
hypothesis = {
    'maingloss': [],
    'domgloss': [],
    'ndomgloss': []
}

accuracies = {}
model.eval()
with torch.no_grad():
    for batch in dataselection.val_dataloader:
        file_Id, text_tokens_padded, gloss_tokens_padded, sample_framestart, boolean_features_padded, inflection_padded = batch
        text_tokens_padded = text_tokens_padded.to(device)
        gloss_tokens_padded = {k: v.to(device) for k, v in gloss_tokens_padded.items()}
        boolean_features_padded = {k: v.to(device) for k, v in boolean_features_padded.items()}
        sample_framestart = sample_framestart.to(device)
        inflection_padded = {k: v.to(device) for k, v in inflection_padded.items()}

        gloss_logits, sample_framestart_values, boolean_pred, inflection_preds = model(text_tokens_padded)
        for gloss_type in ['maingloss', 'domgloss', 'ndomgloss']:
            if gloss_type in gloss_logits:
                gloss_pred = torch.argmax(gloss_logits[gloss_type], dim=-1)

                for i in range(text_tokens_padded.size(0)):
                    text_tokens = text_tokens_padded[i].cpu().numpy()
                    text_tokens = text_tokens[text_tokens != 0]  # Remove padding tokens
                    input_text = tokenizer.decode(text_tokens, skip_special_tokens=True)
                   
                    # Process ground truth maingloss
                    gt_gloss = gloss_tokens_padded[gloss_type][i].cpu().numpy()
                    gt_gloss = gt_gloss[gt_gloss != 0]  # Remove padding tokens
                    gt_gloss_text = "".join([tokenizer.decode([token], skip_special_tokens=True) for token in gt_gloss])
                   
                    # Process predicted maingloss
                    pred_gloss = gloss_pred[i].cpu().numpy()
                    pred_gloss = pred_gloss[pred_gloss != 0]  # Remove padding tokens
                    max_length = 50  # or whatever is appropriate for your task
                    pred_gloss = pred_gloss[:max_length]
                    pred_gloss_text = "".join([tokenizer.decode([token], skip_special_tokens=True) for token in pred_gloss])
                    if gloss_type == 'maingloss':
                        print("\n")
                        print("Text Tokens (input):", input_text)
                        print("Maingloss Tokens (ground truth):", gt_gloss_text)                    
                        print("Maingloss Tokens (prediction):", pred_gloss_text)

                
                ground_truth[gloss_type].append(gt_gloss_text)
                hypothesis[gloss_type].append(pred_gloss_text)


        for feature, pred in inflection_preds.items():
            if inflection_stats[feature] is None:
                continue  # Skip features with no valid statistics

            gt = inflection_padded[feature].to(device)
            pred = pred[:, :gt.size(1)]  # Trim prediction to match ground truth length
            
            # Calculate MSE for normalized values
            valid_indices = gt != 0
            gt_valid = gt[valid_indices]
            pred_valid = pred[valid_indices]
            mse_loss = F.mse_loss(pred_valid, gt_valid).item()
            mse_losses[feature].append(mse_loss)

            # Denormalize and calculate MSE for denormalized values
            gt_denorm = denormalize_and_remove_padding(gt[0].cpu().numpy(), inflection_stats[feature])
            pred_denorm = denormalize_and_remove_padding(pred[0].cpu().numpy(), inflection_stats[feature])
            pred_denorm = pred_denorm[:len(gt_denorm)]
            mse_loss_denorm = mean_squared_error(gt_denorm, pred_denorm)
            mse_losses_denorm[feature].append(mse_loss_denorm)

        gt_framestart = denormalize_feature(sample_framestart[0].cpu().numpy(),framestart_scaler )
        pred_framestart = denormalize_feature(sample_framestart_values[0].cpu().numpy(), framestart_scaler)
        # Trim predictions to match ground truth length
        pred_framestart = pred_framestart[:len(gt_framestart)]        
        # Create mask for non-padding values
        mask_start = gt_framestart != 0
        print("\n")
        print("Framestart Values (ground truth, denormalized):", gt_framestart[mask_start])
        print("Framestart Values (prediction, denormalized):", pred_framestart[mask_start])
        print("Framestart Difference (ground truth - prediction):", gt_framestart[mask_start] - pred_framestart[mask_start])

        gt_framestart_valid = sample_framestart[0].cpu().numpy()[mask_start]
        pred_framestart_valid = sample_framestart_values[0].cpu().numpy()[:len(gt_framestart_valid)]
        mse_loss_framestart = mean_squared_error(gt_framestart_valid, pred_framestart_valid)
        mse_framestart_values.append(mse_loss_framestart)

        # Calculate MSE for framestart (denormalized)
        mse_loss_framestart_denorm = mean_squared_error(gt_framestart[mask_start], pred_framestart[mask_start])
        mse_framestart_values_denorm.append(mse_loss_framestart_denorm)


        for feature, pred in boolean_pred.items():
            if feature not in accuracies:
                accuracies[feature] = {'correct': 0, 'total': 0}

            gt_feature = boolean_features_padded[feature]
            pred_feature = torch.argmax(pred, dim=-1)

            correct, total = calculate_accuracy(pred_feature, gt_feature)
            accuracies[feature]['correct'] += correct
            accuracies[feature]['total'] += total

for feature, counts in accuracies.items():
    accuracy = counts['correct'] / counts['total'] if counts['total'] > 0 else 0
    print(f"\nOverall {feature.capitalize()} Accuracy: {accuracy:.4f}")

print("\n Average MSE Loss for Denormalized")
for feature in mse_losses.keys():
    if mse_losses[feature]: 
        avg_mse = np.mean(mse_losses[feature])
        avg_mse_denorm = np.mean(mse_losses_denorm[feature])
        # print(f"\nAverage MSE Loss for {feature}: {avg_mse}")
        print(f" {feature}: {avg_mse_denorm}")


print("\n")
avg_mse_framestart_loss = np.mean(mse_framestart_values)
avg_mse_framestart_loss_denorm = np.mean(mse_framestart_values_denorm)
# print(f"Average MSE Loss for Framestart: {avg_mse_framestart_loss}")
print(f"Average MSE Loss for Denormalized Framestart: {avg_mse_framestart_loss_denorm}")


# Calculate BLEU score
from sacrebleu.metrics import BLEU
bleu = BLEU()

# Save results to file
save_folder_address = "source_only/" 
with open(save_folder_address + "validation_outputs.txt", "w") as f:
    for gloss_type, ground_value in ground_truth.items():
        hypo = hypothesis[gloss_type]
        result = bleu.corpus_score(hypo, [ground_value])

        print(f"\nBLEU score for {gloss_type}: {result.score}")

        # Count sequence length comparisons
        num_P_T = sum(len(h.split()) > len(g.split()) for h, g in zip(hypo, ground_value))
        num_T_P = sum(len(h.split()) < len(g.split()) for h, g in zip(hypo, ground_value))
        num_e = sum(len(h.split()) == len(g.split()) for h, g in zip(hypo, ground_value))

        print(f"Predicted length > True length: {num_P_T}")
        print(f"True length > Predicted length: {num_T_P}")
        print(f"Equal lengths: {num_e}")

        # Write results to file
        f.write(f"{gloss_type}:\n")
        f.write(f"P>T: {num_P_T}\n")
        f.write(f"T>P: {num_T_P}\n")
        f.write(f"equal: {num_e}\n")
        f.write(f"BLEU score: {result.score}\n\n")

        for i, (gt, pred) in enumerate(zip(ground_value, hypo)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"Ground Truth {gloss_type}: {gt}\n")
            f.write(f"Predicted {gloss_type}: {pred}\n\n")

print(f"Detailed results saved to {save_folder_address}validation_outputs.txt")