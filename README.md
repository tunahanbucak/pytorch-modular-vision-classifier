# 🧁 PyTorch Modular Vision Classifier (Desert101)

Bu proje, keşifsel Jupyter Notebook yapısından çıkılarak **üretim standartlarında modüler Python (`.py`) mimarisiyle** geliştirilmiş bir **PyTorch Evrişimli Sinir Ağı (CNN)** görüntü sınıflandırma projesidir.

Proje; **Desert101** (Food-101 alt kümesi: `baklava`, `cannoli`, `cup_cakes`, `donuts`) veri seti üzerinde uçtan uca veri yükleme, veri artırma (Data Augmentation), kanal normalizasyonu, model eğitimi, ağırlık kaydetme ve tekil görsel çıkarımı (Inference) aşamalarını bağımsız modüller halinde sunar.

---

## 🏗️ Proje Mimarisi ve Dosya Yapısı

```text
├── main.py                         # Ana orkestrasyon dosyası (hiperparametreler, eğitim & kayıt)
├── setup_data.py                   # ImageFolder ve DataLoader kurulumu (NUM_WORKERS desteği)
├── model_creation.py               # TinyVGG tabanlı 2 bloklu DesertClassifier CNN mimarisi
├── training_testing_engine.py      # Modüler train_step(), test_step() ve train() motoru
├── utils.py                        # Model kaydetme ve veri seti Mean / Std hesaplama araçları
├── load_model_make_prediction.py   # Eğitilmiş .pth modelini yükleyip harici görsel tahmin etme
├── models/
│   └── desert_classifier.pth       # Eğitilmiş model ağırlıkları (state_dict)
├── requirements.txt                # Proje bağımlılıkları
└── .gitignore                      # IDE (.idea), sanal ortam (.venv) ve ham veri seti hariç tutma
```

---

## 🧠 Teknik Özellikler ve Overfitting Çözüm Stratejileri

1. **Modüler Yazılım Mimarisi (Separation of Concerns):**
   * Veri hattı, model mimarisi, eğitim/test motoru ve yardımcı araçlar ayrı dosyalara bölünerek yeniden kullanılabilirlik sağlanmıştır.
   * `torch.multiprocessing.set_start_method("spawn", force=True)` ile macOS/Windows ortamlarında çoklu işlem (`num_workers`) uyumluluğu sağlanmıştır.
2. **Veri Artırma (Data Augmentation):**
   * Küçük ölçekli görsel veri setinde ezberlemeyi (**Overfitting**) önlemek için `RandomHorizontalFlip(p=0.4)` ve `TrivialAugmentWide()` uygulanmıştır.
3. **Veri Setine Özel Normalizasyon:**
   * `utils.py` içindeki `get_mean_and_std()` fonksiyonu ile Desert101 eğitim setinin gerçek kanal ortalamaları ve standart sapmaları hesaplanmıştır:
     * `Mean`: `(0.5483, 0.4638, 0.3865)`
     * `Std`: `(0.2173, 0.2279, 0.2263)`
4. **CNN Mimarisi (`DesertClassifier`):**
   * İki ardışık evrişim bloğu (`Conv2d(3x3, pad=1)` $\to$ `ReLU` $\to$ `Conv2d(3x3, pad=1)` $\to$ `ReLU` $\to$ `MaxPool2d(2)`) ile $64 \times 64$ girişler $16 \times 16$ özellik haritalarına indirgenir.
   * Sınıflandırıcı katmanı: `Flatten()` $\to$ `Linear(hidden_units * 16 * 16, output_shape)`.

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Ortam Kurulumu
```bash
git clone https://github.com/tunahanbucak/<REPO-ADI>.git
cd <REPO-ADI>

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Veri Seti Klasör Yapısı
Ham veri seti boyut nedeniyle repoda tutulmamaktadır. Çalıştırmak için görselleri aşağıdaki hiyerarşide `data/` klasörüne yerleştirin:
```text
data/
└── desert101/
    ├── train/
    │   ├── baklava/
    │   ├── cannoli/
    │   ├── cup_cakes/
    │   └── donuts/
    └── test/
        ├── baklava/
        ├── cannoli/
        ├── cup_cakes/
        └── donuts/
```

### 3. Modeli Eğitme (`main.py`)
Varsayılan hiperparametrelerle (`NUM_EPOCHS=10`, `BATCH_SIZE=32`, `HIDDEN_UNITS=32`, `LEARNING_RATE=0.001`) eğitimi başlatmak ve modeli `models/desert_classifier.pth` olarak kaydetmek için:
```bash
python main.py
```

### 4. Kaydedilmiş Model ile Tahmin Yapma (`load_model_make_prediction.py`)
`data/baklava-online.jpg` yolundaki tekil görseli okuyup sınıf tahmini almak için:
```bash
python load_model_make_prediction.py
```
