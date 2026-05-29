# -*- coding: utf-8 -*-
"""
Created on Tue May 26 21:25:49 2026

@author: SELENAY
"""

import torch
import pandas as pd
import numpy as np
import time
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback,
                          BitsAndBytesConfig)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Cihaz: {device}")

train = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\train.csv")
val   = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\val.csv")
test  = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\test.csv")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["full_text"], truncation=True, max_length=128, padding="max_length")

train_ds = Dataset.from_pandas(train).map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val).map(tokenize, batched=True)
test_ds  = Dataset.from_pandas(test).map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
val_ds   = val_ds.rename_column("label", "labels")
test_ds  = test_ds.rename_column("label", "labels")
train_ds.set_format("torch")
val_ds.set_format("torch")
test_ds.set_format("torch")

# 4-bit QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.pad_token_id = tokenizer.pad_token_id

# LoRA config
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

training_args = TrainingArguments(
    output_dir="./tinyllama_qlora",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=1.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    logging_steps=50,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("TinyLlama-1.1B QLoRA başlıyor...")
torch.cuda.reset_peak_memory_stats()
start = time.time()
trainer.train()
train_time = time.time() - start
print(f"Eğitim süresi: {train_time/60:.1f} dakika")

preds_out = trainer.predict(test_ds)
logits = preds_out.predictions
labels = preds_out.label_ids
preds = np.argmax(logits, axis=-1)
probs = torch.softmax(torch.tensor(logits.astype(np.float32)), dim=-1)[:,1].numpy()

acc  = accuracy_score(labels, preds)
f1   = f1_score(labels, preds, average="macro")
prec = precision_score(labels, preds)
rec  = recall_score(labels, preds)
auc  = roc_auc_score(labels, probs)
vram = torch.cuda.max_memory_allocated() / 1024**2

print(f"\n--- TinyLlama-1.1B QLoRA Sonuçları ---")
print(f"Accuracy:   {acc:.4f}")
print(f"Macro-F1:   {f1:.4f}")
print(f"Precision:  {prec:.4f}")
print(f"Recall:     {rec:.4f}")
print(f"ROC-AUC:    {auc:.4f}")
print(f"Max VRAM:   {vram:.0f} MB")
print(f"Süre:       {train_time/60:.1f} dakika")

model.save_pretrained("./tinyllama_qlora_final")
tokenizer.save_pretrained("./tinyllama_qlora_final")
print("Model kaydedildi.")