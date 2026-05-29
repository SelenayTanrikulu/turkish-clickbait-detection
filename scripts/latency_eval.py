# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:17:38 2026

@author: SELENAY
"""

import torch
import time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

device = "cuda"
test = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\test.csv")
texts = test["full_text"].tolist()[:500]  # 500 örnek yeterli

def measure_latency(model, tokenizer, texts, label):
    model.eval()
    latencies = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        start = time.time()
        with torch.no_grad():
            model(**inputs)
        latencies.append((time.time() - start) * 1000)
    avg = sum(latencies) / len(latencies)
    print(f"{label} Avg Latency: {avg:.2f} ms")
    return avg

# SmolLM2
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\SELENAY\smollm2_full_ft_final")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\SELENAY\smollm2_full_ft_final").to(device)
measure_latency(model, tokenizer, texts, "SmolLM2-360M")
del model; torch.cuda.empty_cache()

# TinyLlama
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\SELENAY\tinyllama_qlora_final")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\SELENAY\tinyllama_qlora_final").to(device)
measure_latency(model, tokenizer, texts, "TinyLlama-1.1B")
del model; torch.cuda.empty_cache()

# Qwen
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\SELENAY\qwen_qlora_final")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(r"C:\Users\SELENAY\qwen_qlora_final").to(device)
measure_latency(model, tokenizer, texts, "Qwen2.5-1.5B")
del model; torch.cuda.empty_cache()