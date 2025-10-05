"""
Assignment 1: Data Collection and Preprocessing
Combined implementation of:
1. Wikipedia data downloading with streaming
2. Text cleaning and normalization
3. BERT tokenization
4. PyTorch DataLoader implementation
"""

import os
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

def download_wikipedia():
    """Streaming download with precise 1GB control"""
    stream = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True
    )

    target_bytes = 1 * 1024 ** 3
    current_bytes = 0
    samples = []

    with tqdm(desc="Downloading", unit="MB") as progress:
        for sample in stream:
            text = sample["text"]
            sample_bytes = len(text.encode('utf-8'))

            if current_bytes + sample_bytes > target_bytes:
                break

            samples.append(text)
            current_bytes += sample_bytes
            progress.update(sample_bytes / 1024 ** 2)

    print(f"Downloaded: {current_bytes / 1024 ** 3:.2f}GB ({len(samples)} documents)")
    return samples

def preprocess_texts(texts, min_words=50):
    """Text cleaning pipeline with deduplication and filtering"""
    unique_texts = list(set(texts))

    cleaned = []
    for text in unique_texts:
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        if len(text.split()) >= min_words:
            cleaned.append(text)

    print(f"Cleaned texts: {len(cleaned)}/{len(texts)} retained")
    return cleaned

def tokenize_texts(texts):
    """BERT tokenization with sequence handling"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

class ProcessedDataset(Dataset):
    """Custom PyTorch Dataset for efficient batch processing"""
    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx]
        }

def create_dataloader(encodings, batch_size=32):
    """DataLoader with batching and shuffling support"""
    return DataLoader(
        ProcessedDataset(encodings),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

def save_tokenized_batches(dataloader, num_batches=5, output_file="sample_dataset.pt"):
    """Save first 5-10 batches of tokenized data"""
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        batches.append(batch)

    torch.save(batches, output_file)
    return batches

def main():
    # Data collection
    print("Downloading Wikipedia data")
    texts = download_wikipedia()

    # Preprocessing
    print("Cleaning and normalizing text")
    cleaned_texts = preprocess_texts(texts)

    # Tokenization
    print("Tokenizing with BERT tokenizer")
    tokenized = tokenize_texts(cleaned_texts)

    # Data loader
    print("Creating PyTorch DataLoader")
    dataloader = create_dataloader(tokenized)

    # Save sample batches
    print("Saving tokenized batches")
    sample_batches = save_tokenized_batches(dataloader, num_batches=5)

    print("Processing complete")
    print(f"Batch structure: {sample_batches[0]['input_ids'].shape}")
    print(f"Total batches saved: {len(sample_batches)}")

if __name__ == "__main__":
    main()