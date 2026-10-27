# Türkçe Haber Başlıklarında Clickbait Tespiti

Bu proje, Türkçe haber başlıklarında clickbait tespiti yapmak için SmolLM2, TinyLlama ve Qwen2.5 modellerinin fine-tuning edildiği bir çalışmadır.

## 📁 Proje Yapısı
- `scripts/` → Tüm eğitim ve değerlendirme kodları
- `models/` → Eğitilmiş modeller (LoRA + Full FT)

## Seed Değerleri

•	Train/val/test split: random_state=42
•	Dengeli örnekleme: random_state=42
•	Trainer: seed=42 (TrainingArguments varsayılanı)

## Sonuçlar

Test seti: 4.806 örnek (eğitim verisi: 48.060 Türkçe haber başlığı, dengeli, CC BY-SA 4.0)

| Model | Yöntem | Macro-F1 | Zero-shot F1 | Inference VRAM |
|---|---|---|---|---|
| SmolLM2-360M | Full FT | 0,969 | 0,333 | 1,4 GB |
| TinyLlama-1.1B | QLoRA | 0,979 | 0,408 | 2,7 GB |
| Qwen2.5-1.5B | QLoRA | **0,981** | 0,334 | 3,7 GB |


## 🛠 Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/kullanıcı-adın/turkish-clickbait-detection.git
cd turkish-clickbait-detection

# 2. Sanal ortam oluştur (önerilen)
python -m venv venv
venv\Scripts\activate

# 3. Gereksinimleri yükle
pip install -r requirements.txt
