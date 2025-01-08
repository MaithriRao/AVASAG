import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pickle
import os
from sacrebleu.metrics import BLEU
from pathlib import Path
from torch.utils.data import DataLoader
import time
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import bitsandbytes as bnb
import transformers
import json
import pandas as pd
from datasets import Dataset, load_dataset
from .utils import Translation


def evaluation(translation):

    if translation == Translation.TextToGloss:
        translation_dir = "t2g_llama"
    elif translation == Translation.GlossToText:
        translation_dir = "g2t_llama"
    else:
        raise ValueError("Invalid translation")

    folder_path = os.path.join("/ds/videos/AVASAG/llama_finetune/", translation_dir)
    merged_model_name = os.path.join(folder_path, "llama-31-it-8b-sft-merged")
    cache_dir = "/ds/videos/AVASAG/cache"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_finetune = AutoModelForCausalLM.from_pretrained(
        merged_model_name,
        local_files_only=True,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer_finetune = AutoTokenizer.from_pretrained(
        merged_model_name,
        local_files_only=True,
        add_eos_token=True)


    with open(f'test_data_{translation_dir}.json', 'r') as f:
        test_data = json.load(f)

    # Initialize BLEU metric
    bleu = BLEU()
    references = []
    predictions = []

    # Loop through the test data and generate translations
    for entry in test_data:
        # Extract the text before and after ###>
        my_text = entry["text"].split("###>")[0].strip()
        prompt = my_text+" ###>"
        assert entry["text"].startswith(prompt), f"Prompt not found in the text: {entry['text']}"
        reference = entry["text"].split("###>")[1].strip()
        print("Input is:", my_text)
        print("Ground truth is:", reference)

        # Tokenize and generate the translation
        tokenized_input = tokenizer_finetune(prompt, return_tensors="pt")
        input_ids = tokenized_input["input_ids"].cuda()
        attention_mask = tokenized_input["attention_mask"].cuda()
        reference_length = len(tokenizer_finetune(reference)["input_ids"])  # Get the number of tokens in reference


        # Generate the translation using the model
        generation_output = model_finetune.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=6,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens= reference_length
        )

        # Decode the generated output
        for seq in generation_output.sequences:
            output = tokenizer_finetune.decode(seq, skip_special_tokens=True).split("###>")[1].strip()
            predictions.append(output)
            print("Generated output:", output)
            print("\n")

        # Append the reference to the references list
        references.append([reference])

    # Calculate BLEU score
    bleu_score = bleu.corpus_score(predictions, references)

    # Print the BLEU score
    print(f"BLEU Score: {bleu_score.score}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python k_fold.py [--textTogloss|--glossTotext]")
        sys.exit(1)

    if sys.argv[1] == "--textTogloss":
        print("Translating from Text to  Gloss")
        translation = Translation.TextToGloss
    elif sys.argv[1] == "--glossTotext":
        print("Translating from Gloss to Text ")
        translation = Translation.GlossToText
    else:
        print("You have to specify either --textTogloss or --glossTotext as an argument.")
        sys.exit(1)

    evaluation(translation)
