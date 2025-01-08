import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import os
from sacrebleu.metrics import BLEU
from .data_selection import *
from pathlib import Path
from torch.utils.data import DataLoader
import time
from enum import Enum, verify, UNIQUE
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

hf_access_token = os.getenv("HF_ACCESS_TOKEN")
assert hf_access_token is not None, "You need to set the Hugging Face access token environment variable: export HF_ACCESS_TOKEN=hf_TODO"

login(token = hf_access_token)

def training(translation):

    create_datasets(translation)

    if translation == Translation.TextToGloss:
        translation_dir = "t2g_llama"
    elif translation == Translation.GlossToText:
        translation_dir = "g2t_llama"
    else:
        raise ValueError("Invalid translation")


    with open(f"train_data_{translation_dir}.json", "r") as f:
        train_data = json.load(f)

    with open(f"val_data_{translation_dir}.json", "r") as f:
        val_data = json.load(f)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    cache_dir = "/ds/videos/AVASAG/cache"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_access_token, cache_dir=cache_dir, add_eos_token=True)
    # Set padding token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    save_folder = os.path.join("/ds/videos/AVASAG/llama_finetune/", translation_dir)
    sft_model_name = os.path.join(save_folder, "llama-31-it-8b-sft")
    merged_model_name=os.path.join(save_folder, "llama-31-it-8b-sft-merged")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config, token=hf_access_token, cache_dir=cache_dir)

    model = prepare_model_for_kbit_training(model)

    modules = ["down_proj","up_proj","gate_proj"]

    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    tokenizer.pad_token = tokenizer.eos_token
    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        peft_config=lora_config,
        args=transformers.TrainingArguments(
            report_to=[],  # Disable logging
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_ratio=0.03,
            max_steps=1000,
            learning_rate=2e-5,
            logging_steps=1,
            output_dir="/ds/videos/AVASAG/llama_finetune/outputs_{translation_dir}",
            optim="paged_adamw_8bit",
            save_strategy="epoch",
            ddp_find_unused_parameters=False,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()

    trainer.model.save_pretrained(sft_model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    merged_model = PeftModel.from_pretrained(base_model, sft_model_name)
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(merged_model_name, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_name)


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

    training(translation)
