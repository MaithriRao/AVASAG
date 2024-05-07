import os
import pickle
from pathlib import Path
from . import data_selection
from .utils import Tokenization

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

def read(tokenization):
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

    print("Training tokenizers for the originl data")
    tokenizer_original = data_selection.train_tokenizers(data_list_only_original)
    (text_tokenizer_original, mms_tokenizer_original) = tokenizer_original

    print("Training tokenizers for the full data")
    tokenizer_full = data_selection.train_tokenizers(data_list_full)
    (text_tokenizer_full, mms_tokenizer_full) = tokenizer_full

    tokenizer = (tokenizer_original, tokenizer_full)

    #this is to use for creating the new data in test_file
    sentences_modified = data_selection.split_dataset(data_list_only_modified)
    (source_text_modified, target_gloss_modified) = sentences_modified

    #vocabulary for the original data
    sentences_original = data_selection.split_dataset(data_list_only_original)
    (source_text_original, target_gloss_original) = sentences_original
    text_vocab_original = data_selection.build_text_vocab(source_text_original, text_tokenizer_original)
    if tokenization == Tokenization.SOURCE_TARGET:
        mms_vocab_original = data_selection.build_mms_vocab_with_tokenizer(target_gloss_original, mms_tokenizer_original)
    elif tokenization == Tokenization.SOURCE_ONLY:
        mms_vocab_original = data_selection.build_mms_vocab_without_tokenizer(target_gloss_original)
    else:
        raise ValueError("Invalid tokenization value")
    vocab_original = (text_vocab_original, mms_vocab_original)

    #vocabulary for the combined data
    sentences_full = data_selection.split_dataset(data_list_full)
    (source_text_full, target_gloss_full) = sentences_full
    text_vocab_full = data_selection.build_text_vocab(source_text_full, text_tokenizer_full)
    if tokenization == Tokenization.SOURCE_TARGET:
        mms_vocab_full = data_selection.build_mms_vocab_with_tokenizer(target_gloss_full, mms_tokenizer_full)
    elif tokenization == Tokenization.SOURCE_ONLY:
        mms_vocab_full = data_selection.build_mms_vocab_without_tokenizer(target_gloss_full)
    else:
        raise ValueError("Invalid tokenization value")
    vocab_full = (text_vocab_full, mms_vocab_full)

    original = (tokenizer_original, vocab_original, sentences_original)
    modified = (sentences_modified,)
    full = (tokenizer_full, vocab_full, sentences_full)

    return (original, modified, full)
