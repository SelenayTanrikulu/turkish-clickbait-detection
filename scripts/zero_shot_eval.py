# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:29:38 2026

@author: SELENAY
"""

import torch
import pandas as pd
import numpy as np
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
test = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\test.csv")
texts = test["full_text"].tolist()
labels = test["label"].tolist()

def zero_shot_eval(model_name, display_name):
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    model.eval()

    preds, probs_list, latencies = [], [], []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        latencies.append((time.time() - start) * 1000)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=-1)[0][1].item()
        probs_list.append(prob)
        preds.append(torch.argmax(logits, dim=-1).item())

    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="macro")
    prec = precision_score(labels, preds)
    rec  = recall_score(labels, preds)
    auc  = roc_auc_score(labels, probs_list)
    lat  = sum(latencies) / len(latencies)
    vram = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n--- {display_name} Zero-Shot ---")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Macro-F1:   {f1:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    print(f"Avg Latency:{lat:.2f} ms")
    print(f"Max VRAM:   {vram:.0f} MB")

    del model
    torch.cuda.empty_cache()

zero_shot_eval("HuggingFaceTB/SmolLM2-360M", "SmolLM2-360M")
zero_shot_eval("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "TinyLlama-1.1B")
zero_shot_eval("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B")