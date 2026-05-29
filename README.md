# Türkçe Haber Başlıklarında Clickbait Tespiti

Bu proje, Türkçe haber başlıklarında clickbait tespiti yapmak için SmolLM2, TinyLlama ve Qwen2.5 modellerinin fine-tuning edildiği bir çalışmadır.

## 📁 Proje Yapısı
- `scripts/` → Tüm eğitim ve değerlendirme kodları
- `models/` → Eğitilmiş modeller (LoRA + Full FT)

## Seed Değerleri

•	Train/val/test split: random_state=42
•	Dengeli örnekleme: random_state=42
•	Trainer: seed=42 (TrainingArguments varsayılanı)


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