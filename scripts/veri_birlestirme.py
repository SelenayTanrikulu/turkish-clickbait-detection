# -*- coding: utf-8 -*-
"""
Türkçe Tıklama Tuzağı Veri Seti — Veri Hazırlama
--------------------------------------------------
Adımlar:
  1. 4 CSV kaynağını yükle ve etiketle
  2. Null temizleme
  3. Train / Validation / Test bölme (%80 / %10 / %10)
"""
 
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
 
# ── Dosya yolları ──────────────────────────────────────────────────────────────
# Scriptin bulunduğu klasörü kök olarak al;
# CSV'leri buraya göre konumlandır.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)
 
# ── 1. Veri setlerini yükle ────────────────────────────────────────────────────
limon   = pd.read_csv(DATA_DIR / "limon_clickbait.csv")
spoiler = pd.read_csv(DATA_DIR / "spoiler_clickbait.csv")
diken   = pd.read_csv(DATA_DIR / "diken_non-clickbait.csv")
evrensel = pd.read_csv(DATA_DIR / "evrensel_non-clickbait.csv")
 
# Etiket ekle (1 = clickbait, 0 = non-clickbait)
limon["label"]    = 1
spoiler["label"]  = 1
diken["label"]    = 0
evrensel["label"] = 0
 
# Birleştir
df = pd.concat([limon, spoiler, diken, evrensel], ignore_index=True)
print("── Ham veri ──────────────────────────────")
print(f"Boyut   : {df.shape}")
print(f"Kolonlar: {df.columns.tolist()}")
print(df["label"].value_counts())
print(df.head(3))
 
# ── 2. Null temizleme ──────────────────────────────────────────────────────────
print(f"\n── Null temizleme ────────────────────────")
print(f"Null satır sayısı: {df['full_text'].isnull().sum()}")
 
df = df[["full_text", "label"]].dropna().reset_index(drop=True)
print(f"Temizlenmiş boyut: {df.shape}")
print(df["label"].value_counts())
 
df.to_csv(DATA_DIR / "dataset.csv", index=False)
print("dataset.csv kaydedildi.")
 
# ── 3. Train / Validation / Test bölme (%80 / %10 / %10) ──────────────────────
print(f"\n── Veri bölme (%80 / %10 / %10) ─────────")
 
# Önce %10 test ayır
train_val, test = train_test_split(
    df, test_size=0.10, random_state=42, stratify=df["label"]
)
# Kalanı %80 train, %10 validation olarak böl (~%11.1 of train_val)
train, val = train_test_split(
    train_val, test_size=0.111, random_state=42, stratify=train_val["label"]
)
 
print(f"Train      : {len(train)}")
print(f"Validation : {len(val)}")
print(f"Test       : {len(test)}")
 
train.to_csv(DATA_DIR / "train.csv", index=False)
val.to_csv(DATA_DIR  / "val.csv",   index=False)
test.to_csv(DATA_DIR / "test.csv",  index=False)
print("train.csv / val.csv / test.csv kaydedildi.")