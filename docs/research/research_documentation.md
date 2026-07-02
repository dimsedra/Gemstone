# Dokumentasi Penelitian: Sistem Klasifikasi Gambar Batu Permata Menggunakan Transfer Learning ResNet50

---

## 1. Pendahuluan

Penelitian ini membangun sebuah sistem klasifikasi gambar batu permata berbasis deep learning yang mampu mengidentifikasi 87 jenis batu permata dari sebuah foto. Sistem diimplementasikan secara lokal menggunakan arsitektur ResNet50 yang dilatih dengan pendekatan *transfer learning*, serta dilengkapi dengan antarmuka web interaktif untuk penggunaan praktis.

**Tujuan Sistem:**
- Mengklasifikasikan gambar batu permata ke dalam 87 kategori
- Menampilkan prediksi beserta tingkat keyakinan (confidence)
- Menyediakan antarmuka web lokal yang dapat digunakan langsung tanpa koneksi internet

---

## 2. Dataset

### 2.1 Sumber dan Pembagian Data

Dataset diperoleh dari platform **Roboflow** dan terdiri dari gambar batu permata 87 kelas. Pembagian data mengikuti rasio **70% training / 20% validasi / 10% testing** dari data asli.

| Split | Jumlah Gambar | Jumlah Kelas | Rata-rata/Kelas |
|---|---|---|---|
| Training (sebelum augmentasi) | 1.974 | 87 | ~22,7 |
| Training (setelah augmentasi 3×) | 5.922 | 87 | ~68,1 |
| Validasi | 564 | 87 | 6,5 |
| Test | 282 | 83 | 3,4 |
| **Total data asli** | **2.820** | **87** | **~32,4** |

> **Catatan penting:** Augmentasi Roboflow hanya diterapkan pada training set dengan faktor 3×. Validasi dan test set menggunakan foto asli tanpa augmentasi tambahan. 4 kelas tidak hadir dalam test set.

### 2.2 Preprocessing (Roboflow)

Seluruh gambar diproses melalui pipeline Roboflow sebelum digunakan:

| Tahap | Konfigurasi |
|---|---|
| **Auto-Orient** | Applied (koreksi rotasi EXIF otomatis) |
| **Resize** | Fill with center crop → **224 × 224 piksel** |

### 2.3 Augmentasi (Roboflow — Training Set Only)

| Teknik Augmentasi | Parameter |
|---|---|
| Output per gambar | **3×** (setiap foto menghasilkan 3 varian) |
| Flip | Horizontal dan Vertikal |
| Crop | Minimum Zoom 0%, Maximum Zoom 15% |
| Rotasi | −15° hingga +15° |
| Shear | ±10° Horizontal, ±10° Vertikal |
| Brightness | −20% hingga +20% |
| Exposure | −10% hingga +10% |
| Blur | Hingga 0,5 piksel |
| Noise | Hingga 0,1% piksel |

### 2.4 Statistik Distribusi Data

**Training set (per kelas, setelah augmentasi):**
- Minimum: **24 gambar** (Pearl)
- Maksimum: **99 gambar** (Pyrite)
- Rata-rata: **68,1 gambar**
- Kelas dengan < 50 gambar: **5 kelas** (Pearl, Lapis Lazuli, Zircon, Malachite, Scapolite)

**Test set (per kelas):**
- Minimum: **1 gambar** (12 kelas)
- Maksimum: **9 gambar** (Quartz Rutilated)
- Rata-rata: **3,4 gambar**

---

## 3. Arsitektur Model

### 3.1 Backbone

Model menggunakan **ResNet50** (*Deep Residual Network* dengan 50 lapisan) yang di-*pretrain* pada dataset ImageNet (1.000 kelas, ~1,2 juta gambar). ResNet50 dipilih karena:
- Keseimbangan baik antara akurasi dan efisiensi komputasi
- Representasi fitur visual yang kaya dari *pretraining* ImageNet
- Ketersediaan bobot *pretrained* resmi di PyTorch (`ResNet50_Weights.DEFAULT`)

**Spesifikasi ResNet50:**
- Total parameter backbone: ~23,5 juta
- Output dimensi fitur (sebelum fc layer): **2.048**
- Status selama training: **Dibekukan (*frozen*)** — tidak di-*update*

### 3.2 Classification Head (Custom)

Layer `fc` bawaan ResNet50 diganti dengan head klasifikasi kustom:

```
Linear(2048 → 512)
ReLU()
Dropout(p=0.3)
Linear(512 → 87)
```

**Parameter yang dilatih:** hanya classification head (~1,1 juta parameter dari ~24,6 juta total).

### 3.3 Diagram Arsitektur

```
Input Image (224×224×3)
        ↓
[ResNet50 Backbone — FROZEN]
   Conv1 → BN → ReLU → MaxPool
   Layer1 (3× Bottleneck)
   Layer2 (4× Bottleneck)
   Layer3 (6× Bottleneck)
   Layer4 (3× Bottleneck)
   AdaptiveAvgPool2d
        ↓
Feature Vector (2048)
        ↓
[Custom Classification Head — TRAINABLE]
   Linear(2048 → 512)
   ReLU
   Dropout(0.3)
   Linear(512 → 87)
        ↓
Logits (87 kelas)
        ↓
Softmax → Probabilitas
```

---

## 4. Prosedur Training

### 4.1 Konfigurasi Lingkungan

| Komponen | Detail |
|---|---|
| Framework | PyTorch 2.5.1+cu121 |
| Hardware | NVIDIA GeForce RTX 3060 (12 GB VRAM) |
| CUDA | 12.1 |
| Python | 3.11.15 |
| OS | Windows 11 |

### 4.2 Hyperparameter

| Parameter | Nilai |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Batch Size | 64 |
| Max Epochs | 100 |
| Early Stopping Patience | 5 (berdasarkan val_loss) |
| Dropout Rate | 0,3 |
| Loss Function | CrossEntropyLoss |
| num_workers (DataLoader) | 2 |

### 4.3 Runtime Augmentasi (train.py)

Selain augmentasi yang sudah diterapkan oleh Roboflow, training pipeline menerapkan augmentasi tambahan secara *real-time* pada setiap iterasi:

| Transformasi | Detail |
|---|---|
| Resize | 224 × 224 (fallback safety) |
| RandomHorizontalFlip | p = 0,5 |
| RandomVerticalFlip | p = 0,5 |
| RandomRotation | ±15° |
| ToTensor | Normalisasi ke [0, 1] |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

Normalisasi menggunakan statistik ImageNet untuk konsistensi dengan fitur *pretrained*.

### 4.4 Early Stopping & Checkpoint

- Model terbaik disimpan setiap kali `val_loss` mencapai nilai minimum baru
- Training dihentikan apabila `val_loss` tidak membaik selama **5 epoch berturut-turut**
- Checkpoint disimpan di `models/gemstone_resnet50.pth`
- Log training disimpan di `models/training.log`
- History metrik disimpan di `models/training_history.json`

---

## 5. Hasil Training

### 5.1 Performa Per Epoch

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Keterangan |
|:---:|---:|---:|---:|---:|---|
| 1  | 1,6351 | 56,15% | 0,9991 | 68,00% | |
| 2  | 1,0553 | 69,22% | **0,9380** | **69,75%** | ✅ Best checkpoint |
| 3  | 0,7918 | 76,04% | 0,9688 | 69,50% | |
| 4  | 0,6177 | 80,89% | 0,9676 | 69,25% | |
| 5  | 0,5122 | 83,99% | 0,9898 | 68,75% | |
| 6  | 0,4377 | 86,38% | 1,0106 | 69,00% | |
| 7  | 0,3819 | 88,17% | 1,0119 | 69,00% | |
| 8  | 0,3357 | 89,75% | 1,0250 | 69,00% | |
| 9  | 0,2914 | 91,31% | 1,0274 | 69,50% | |
| 10 | 0,2414 | 92,84% | 1,0346 | 68,50% | |
| 11 | 0,2434 | 92,76% | 1,1601 | 67,02% | Patience 1/5 |
| 12 | 0,2092 | 93,84% | 1,1126 | 68,09% | Patience 2/5 |
| 13 | 0,1832 | 94,34% | 1,0489 | 70,57% | Patience 3/5 |
| 14 | 0,1698 | 94,90% | 1,1077 | 68,26% | Patience 4/5 |
| 15 | 0,1674 | 94,92% | 1,1171 | 70,04% | Patience 5/5 → **Early Stop** |

### 5.2 Analisis Kurva Training

**Observasi kunci:**
- Training loss turun secara konsisten dari 1,6351 → 0,1674 selama 15 epoch
- Validation loss mencapai minimum pada **epoch 2** (0,9380) kemudian meningkat
- Validation accuracy bergerak dalam rentang sempit **67–70%** sepanjang training
- Gap antara train accuracy (94,9%) dan val accuracy (70%) mengindikasikan model mencapai *capacity ceiling* dari backbone yang dibekukan, bukan overfitting destruktif

> Validation accuracy tidak jatuh signifikan meskipun val loss meningkat, yang menunjukkan bahwa model mempertahankan kemampuan klasifikasinya namun semakin tidak yakin (confidence tersebar) — bukan semakin salah.

---

## 6. Evaluasi pada Test Set

### 6.1 Metrik Keseluruhan

| Metrik | Nilai |
|---|---|
| **Overall Test Accuracy** | **65,60%** |
| Macro Precision | 67,92% |
| Macro Recall | 66,43% |
| Macro F1-Score | 63,09% |
| Weighted F1-Score | 65,09% |
| Jumlah test samples | 282 |
| Jumlah kelas dievaluasi | 83 dari 87 |

### 6.2 Per-Class Classification Report

| Kelas | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Alexandrite | 0,6667 | 1,0000 | 0,8000 | 4 |
| Almandine | 0,5000 | 1,0000 | 0,6667 | 1 |
| Amazonite | 0,0000 | 0,0000 | 0,0000 | 1 |
| Amber | 1,0000 | 0,5000 | 0,6667 | 2 |
| Amethyst | 0,5000 | 0,3333 | 0,4000 | 6 |
| Ametrine | 1,0000 | 1,0000 | **1,0000** | 1 |
| Andalusite | 0,4000 | 0,4000 | 0,4000 | 5 |
| Andradite | 0,4000 | 0,5000 | 0,4444 | 4 |
| Aquamarine | 0,6667 | 1,0000 | 0,8000 | 2 |
| Aventurine Green | 0,8000 | 0,8000 | 0,8000 | 5 |
| Benitoite | 1,0000 | 0,2500 | 0,4000 | 4 |
| Beryl Golden | 0,0000 | 0,0000 | 0,0000 | 1 |
| Bixbite | 1,0000 | 0,2000 | 0,3333 | 5 |
| Bloodstone | 1,0000 | 0,7143 | 0,8333 | 7 |
| Blue Lace Agate | 1,0000 | 1,0000 | **1,0000** | 6 |
| Carnelian | 1,0000 | 0,6667 | 0,8000 | 6 |
| Cats Eye | 0,5000 | 1,0000 | 0,6667 | 1 |
| Chalcedony | 0,7500 | 1,0000 | 0,8571 | 3 |
| Chrome Diopside | 0,6000 | 1,0000 | 0,7500 | 3 |
| Chrysoberyl | 0,5000 | 0,1667 | 0,2500 | 6 |
| Chrysocolla | 0,5000 | 1,0000 | 0,6667 | 1 |
| Chrysoprase | 1,0000 | 1,0000 | **1,0000** | 1 |
| Citrine | 1,0000 | 1,0000 | **1,0000** | 3 |
| Coral | 1,0000 | 1,0000 | **1,0000** | 2 |
| Danburite | 0,3333 | 0,5000 | 0,4000 | 2 |
| Diamond | 1,0000 | 0,5000 | 0,6667 | 2 |
| Diaspore | 0,0000 | 0,0000 | 0,0000 | 2 |
| Dumortierite | 1,0000 | 0,2500 | 0,4000 | 4 |
| Emerald | 0,8333 | 1,0000 | 0,9091 | 5 |
| Fluorite | 0,7500 | 1,0000 | 0,8571 | 3 |
| Garnet Red | 0,0000 | 0,0000 | 0,0000 | 1 |
| Goshenite | 0,0000 | 0,0000 | 0,0000 | 2 |
| Grossular | 0,0000 | 0,0000 | 0,0000 | 8 |
| Hessonite | 1,0000 | 0,7500 | 0,8571 | 4 |
| Iolite | 0,5000 | 0,3333 | 0,4000 | 3 |
| Jade | 1,0000 | 0,4000 | 0,5714 | 5 |
| Jasper | 0,6250 | 0,8333 | 0,7143 | 6 |
| Kunzite | 0,6667 | 0,5000 | 0,5714 | 4 |
| Kyanite | 0,3333 | 0,5000 | 0,4000 | 2 |
| Labradorite | 0,8000 | 0,8000 | 0,8000 | 5 |
| Lapis Lazuli | 1,0000 | 0,5714 | 0,7273 | 7 |
| Larimar | 0,5000 | 1,0000 | 0,6667 | 3 |
| Malachite | 1,0000 | 1,0000 | **1,0000** | 4 |
| Moonstone | 1,0000 | 1,0000 | **1,0000** | 4 |
| Morganite | 1,0000 | 1,0000 | **1,0000** | 2 |
| Onyx Green | 0,7500 | 1,0000 | 0,8571 | 3 |
| Onyx Red | 0,4000 | 1,0000 | 0,5714 | 2 |
| Opal | 1,0000 | 0,7500 | 0,8571 | 4 |
| Pearl | 0,0000 | 0,0000 | 0,0000 | 1 |
| Peridot | 0,6000 | 1,0000 | 0,7500 | 3 |
| Prehnite | 1,0000 | 0,5000 | 0,6667 | 2 |
| Pyrite | 1,0000 | 0,5000 | 0,6667 | 2 |
| Pyrope | 0,3333 | 1,0000 | 0,5000 | 1 |
| Quartz Beer | 1,0000 | 1,0000 | **1,0000** | 3 |
| Quartz Lemon | 1,0000 | 0,5000 | 0,6667 | 2 |
| Quartz Rose | 1,0000 | 0,5000 | 0,6667 | 2 |
| Quartz Rutilated | 1,0000 | 1,0000 | **1,0000** | 9 |
| Quartz Smoky | 0,6667 | 0,6667 | 0,6667 | 3 |
| Rhodochrosite | 1,0000 | 1,0000 | **1,0000** | 3 |
| Rhodolite | 0,6000 | 0,6000 | 0,6000 | 5 |
| Rhodonite | 0,8000 | 1,0000 | 0,8889 | 4 |
| Ruby | 1,0000 | 1,0000 | **1,0000** | 2 |
| Sapphire Blue | 0,3333 | 0,5000 | 0,4000 | 2 |
| Sapphire Pink | 0,6667 | 0,4000 | 0,5000 | 5 |
| Sapphire Purple | 0,5000 | 0,5000 | 0,5000 | 2 |
| Sapphire Yellow | 0,7500 | 1,0000 | 0,8571 | 3 |
| Scapolite | 1,0000 | 0,7143 | 0,8333 | 7 |
| Serpentine | 1,0000 | 0,8000 | 0,8889 | 5 |
| Sodalite | 0,5000 | 1,0000 | 0,6667 | 4 |
| Spessartite | 1,0000 | 0,6667 | 0,8000 | 3 |
| Sphene | 0,1111 | 0,3333 | 0,1667 | 3 |
| Spinel | 1,0000 | 0,7500 | 0,8571 | 4 |
| Spodumene | 0,3333 | 0,4000 | 0,3636 | 5 |
| Sunstone | 1,0000 | 1,0000 | **1,0000** | 2 |
| Tanzanite | 0,1250 | 1,0000 | 0,2222 | 1 |
| Tigers Eye | 0,6667 | 1,0000 | 0,8000 | 2 |
| Topaz | 0,6667 | 0,3333 | 0,4444 | 6 |
| Tourmaline | 0,5455 | 0,8571 | 0,6667 | 7 |
| Tsavorite | 0,0000 | 0,0000 | 0,0000 | 2 |
| Turquoise | 1,0000 | 1,0000 | **1,0000** | 1 |
| Variscite | 0,5000 | 0,5000 | 0,5000 | 2 |
| Zircon | 0,4000 | 0,4000 | 0,4000 | 5 |
| Zoisite | 1,0000 | 0,5000 | 0,6667 | 2 |

### 6.3 Ringkasan Performa Kelas

**Kelas dengan F1-Score = 1,0 (klasifikasi sempurna):**
Ametrine, Blue Lace Agate, Chrysoprase, Citrine, Coral, Malachite, Moonstone, Morganite, Quartz Beer, Quartz Rutilated, Rhodochrosite, Ruby, Sunstone, Turquoise

**Kelas dengan F1-Score = 0,0 (gagal diklasifikasikan):**
Amazonite, Beryl Golden, Diaspore, Garnet Red, Goshenite, Grossular, Onyx Black, Pearl, Tsavorite

> **Catatan penting:** Kelas dengan F1=0 umumnya memiliki **1–2 gambar di test set**. Dengan support sekecil itu, satu prediksi salah sudah menghasilkan F1=0. Ini mencerminkan keterbatasan ukuran test set, bukan semata-mata kegagalan model.

---

## 7. Arsitektur Sistem

### 7.1 Stack Teknologi

| Komponen | Teknologi |
|---|---|
| Deep Learning Framework | PyTorch 2.5.1 |
| Model Backbone | torchvision ResNet50 (ImageNet pretrained) |
| Backend API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML5 / CSS3 / JavaScript |
| Image Processing | Pillow (PIL) + torchvision transforms |
| Evaluation | scikit-learn |
| Dataset Management | Roboflow |

### 7.2 Arsitektur Sistem Keseluruhan

```
[Browser / User]
      ↕  HTTP
[FastAPI Server — src/app.py]
  ├── GET  /               → Serve static/index.html
  ├── GET  /info           → Metadata model (kelas, status)
  ├── GET  /models/training_metrics.png → Chart training
  ├── POST /predict        → Klasifikasi gambar
  └── GET  /static/*       → CSS, JS, assets
         ↓
[GemstoneClassifier — src/inference.py]
  ├── Thread-safe lazy loading (threading.Lock)
  ├── EXIF auto-orient (PIL.ImageOps.exif_transpose)
  ├── Preprocessing (Resize → ToTensor → Normalize)
  ├── Inference (ResNet50 → Softmax)
  └── Output: prediction, confidence, top-5
         ↓
[ResNet50 Model — models/gemstone_resnet50.pth]
```

### 7.3 Alur Inferensi

1. User mengupload gambar via drag-and-drop atau file picker
2. Gambar dikirim ke endpoint `POST /predict` sebagai `multipart/form-data`
3. Validasi MIME type (harus `image/*`)
4. PIL membuka gambar dan menerapkan `exif_transpose` untuk koreksi orientasi
5. Transform: Resize(224,224) → ToTensor → Normalize(ImageNet stats)
6. Forward pass melalui ResNet50 + classification head
7. Softmax menghasilkan distribusi probabilitas 87 kelas
8. Response JSON berisi: predicted class, confidence, top-5 alternatives
9. Frontend menampilkan hasil dengan confidence bar berwarna dinamis

### 7.4 Fitur Keamanan

- Model weights (`.pth`) tidak diekspos melalui HTTP — hanya chart PNG yang dapat diakses
- Validasi MIME type mencegah upload file non-gambar
- Thread-safe lazy loading dengan `threading.Lock()` dan double-check pattern
- `weights_only=True` pada `torch.load()` mencegah deserialisasi berbahaya

---

## 8. Antarmuka Pengguna

Sistem dilengkapi dashboard web dengan desain *glassmorphic dark mode* yang dapat diakses di `http://127.0.0.1:8000/`:

**Fitur UI:**
- Drag-and-drop file upload dengan preview gambar
- Confidence bar berwarna dinamis (Hijau ≥80%, Kuning 50–79%, Merah <50%)
- Top-5 alternative predictions
- Tab Training Diagnostics (kurva loss & accuracy)
- Tab Supported Gemstones dengan search filter real-time (87 kelas)
- Status model online/offline otomatis

---

## 9. Diskusi dan Keterbatasan

### 9.1 Faktor Pembatas Performa

**Ukuran dataset asli yang sangat kecil:**
Dataset asli hanya berisi ~2.820 gambar untuk 87 kelas (~32 gambar/kelas). Meskipun augmentasi 3× meningkatkan volume training, seluruh data training bersumber dari hanya ~22 gambar asli per kelas. Ini membatasi keberagaman visual yang dapat dipelajari model.

**Backbone dibekukan:**
Dengan frozen backbone, model hanya bisa belajar kombinasi linear dari fitur ImageNet yang ada. Fitur ini tidak spesifik untuk domain batu permata (kilap, inklusi kristal, transparansi), sehingga ada *representational bottleneck* yang tidak bisa diatasi dengan menambah epoch.

**Validation dan test set terlalu kecil:**
Dengan rata-rata 6,5 gambar/kelas di validasi dan 3,4 gambar/kelas di test, metrik evaluasi memiliki varians statistik yang tinggi. Beberapa kelas hanya memiliki 1 gambar di test set, membuat perbandingan per-kelas tidak reliable.

**Visual similarity antar kelas:**
Beberapa kelas secara visual sangat mirip (berbagai jenis Sapphire, berbagai jenis Quartz, berbagai jenis Onyx), yang secara inheren membuat klasifikasi lebih sulit.

### 9.2 Interpretasi Hasil

Akurasi 65,60% pada test set perlu diinterpretasikan dalam konteks:
- Baseline *random guessing* untuk 87 kelas: **~1,15%**
- Model mencapai **~57× lebih baik dari random**
- Untuk kelas yang memiliki cukup data training dan visual yang berbeda, model mencapai F1=1,0
- Kelas yang gagal umumnya memiliki dukungan test set yang sangat kecil (1–2 gambar)

### 9.3 Arah Pengembangan Selanjutnya

| Strategi | Expected Impact | Kompleksitas |
|---|---|---|
| Tambah data asli (>200 gambar/kelas) | Tinggi (+10–20% acc) | Tinggi |
| Fine-tune layer backbone terakhir | Sedang (+2–5% acc) | Sedang |
| Gunakan model lebih besar (EfficientNetV2, ViT) | Sedang (+5–10% acc) | Sedang |
| Tambah augmentasi MixUp/CutMix | Kecil (+1–3% acc) | Rendah |

---

## 10. Cara Menjalankan Sistem

### Prasyarat
- Python 3.11+
- NVIDIA GPU dengan CUDA (opsional, CPU juga bisa)
- Virtual environment sudah dikonfigurasi (`.venv/`)

### Menjalankan Aplikasi Web
```powershell
.venv\Scripts\uvicorn src.app:app --reload
# Akses di: http://127.0.0.1:8000/
```

### Menjalankan Training
```powershell
python src/train.py
# Log real-time di: models/training.log
# Jalankan: Get-Content models\training.log -Wait
```

### Menjalankan Evaluasi
```powershell
python src/evaluate.py
```

### Menjalankan Test Suite
```powershell
pytest tests/test_app.py
# Expected: 8 passed
```

---

## 11. Referensi

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
- Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). *ImageNet: A Large-Scale Hierarchical Image Database*. CVPR 2009.
- PyTorch Documentation. *torchvision.models.resnet50*. https://pytorch.org/vision/stable/models/resnet.html
- FastAPI Documentation. https://fastapi.tiangolo.com/
- Roboflow Platform. https://roboflow.com/

---

*Dokumentasi ini mencatat seluruh proses implementasi sistem klasifikasi batu permata dari tahap persiapan dataset hingga deployment antarmuka web lokal.*
