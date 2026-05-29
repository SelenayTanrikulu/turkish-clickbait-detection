# -*- coding: utf-8 -*-
"""
Created on Fri May 29 20:01:48 2026

@author: SELENAY
"""

# -*- coding: utf-8 -*-
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

device = "cuda"
test = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\test.csv")

def get_errors(base_model_name, model_path, model_label, is_peft=True):
    print(f"\n{model_label} - Hata örnekleri toplanıyor...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    if is_peft:
        base = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        model = PeftModel.from_pretrained(base, model_path).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
    
    model.eval()
    errors = []
    
    for _, row in test.iterrows():
        text = row["full_text"]
        true_label = int(row["label"])
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        
        if pred != true_label:
            errors.append({
                "Başlık": text[:140] + "..." if len(text) > 140 else text,
                "Gerçek": "Clickbait" if true_label == 1 else "Normal",
                "Tahmin": "Clickbait" if pred == 1 else "Normal",
                "Model": model_label,
                "Tip": "FP" if pred == 1 else "FN"
            })
    
    del model
    torch.cuda.empty_cache()
    return errors

# ===================== TÜM MODELLER İÇİN HATA TOPLA =====================
all_errors = []

all_errors.extend(get_errors("HuggingFaceTB/SmolLM2-360M", 
                           r"C:\Users\SELENAY\smollm2_full_ft_final", 
                           "SmolLM2-360M", is_peft=False))

all_errors.extend(get_errors("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                           r"C:\Users\SELENAY\tinyllama_qlora_final", 
                           "TinyLlama-1.1B"))

all_errors.extend(get_errors("Qwen/Qwen2.5-1.5B", 
                           r"C:\Users\SELENAY\qwen_qlora_final", 
                           "Qwen2.5-1.5B"))

# ===================== 10 ÖRNEKLİ GÜZEL TABLO =====================
df = pd.DataFrame(all_errors)

print("\n" + "="*110)
print("                          HATA ANALİZİ TABLOSU - İLK 10 ÖRNEK")
print("="*110)

table = df.head(10)[["Başlık", "Gerçek", "Tahmin", "Model", "Tip"]]
print(table.to_string(index=False))

print(f"\nToplam toplanan hata örneği: {len(df)}")