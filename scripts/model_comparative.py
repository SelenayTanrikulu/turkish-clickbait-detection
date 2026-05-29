# -*- coding: utf-8 -*-
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = "cuda"
test = pd.read_csv(r"C:\Users\SELENAY\turkish-clickbait-dataset\dataset\test.csv")
total_samples = len(test)

print(f"Test seti boyutu: {total_samples} örnek\n")

def analyze_model(base_model_name, model_path, model_label, is_peft=False):
    print(f"{model_label} analiz ediliyor...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    if is_peft:
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
        model = PeftModel.from_pretrained(base, model_path).to(device)
    else:
        # Full Fine-Tuning için
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2).to(device)
    
    model.eval()
    
    fp = 0
    fn = 0
    
    for idx, row in test.iterrows():
        text = row["full_text"]
        true_label = int(row["label"])
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        pred = torch.argmax(outputs.logits, dim=-1).item()
        
        if pred != true_label:
            if pred == 1:
                fp += 1
            else:
                fn += 1
        
        if (idx + 1) % 1000 == 0:
            print(f"  → {idx+1}/{total_samples} işlendi")
    
    total_error = fp + fn
    error_rate = (total_error / total_samples) * 100
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "model": model_label,
        "total_error": total_error,
        "fp": fp,
        "fn": fn,
        "error_rate": error_rate
    }

# ===================== 3 MODELİ ANALİZ ET =====================
results = []

# SmolLM2 → Full Fine-Tuning (Peft değil)
results.append(analyze_model(
    "HuggingFaceTB/SmolLM2-360M",
    r"C:\Users\SELENAY\smollm2_full_ft_final",   # <-- Full FT klasörü
    "SmolLM2-360M",
    is_peft=False
))

# TinyLlama → LoRA
results.append(analyze_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    r"C:\Users\SELENAY\tinyllama_qlora_final",
    "TinyLlama-1.1B",
    is_peft=True
))

# Qwen → LoRA
results.append(analyze_model(
    "Qwen/Qwen2.5-1.5B",
    r"C:\Users\SELENAY\qwen_qlora_final",
    "Qwen2.5-1.5B",
    is_peft=True
))

# ===================== TABLO =====================
print("\n" + "="*85)
print("                    MODEL KARŞILAŞTIRMA TABLOSU")
print("="*85)

df = pd.DataFrame(results)
df = df[["model", "total_error", "fp", "fn", "error_rate"]]
df.columns = ["Model", "Toplam Hata", "FP", "FN", "Hata Oranı (%)"]
df["Hata Oranı (%)"] = df["Hata Oranı (%)"].map("{:.2f}".format)

print(df.to_string(index=False))

best = df.loc[df["Toplam Hata"].idxmin()]
print(f"\n⭐ En iyi model: {best['Model']} ({best['Toplam Hata']} hata - %{best['Hata Oranı (%)']})")