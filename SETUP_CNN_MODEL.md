# Setup CNN Model Files

Untuk menggunakan fitur CNN detection, Anda perlu download 2 file model MobileNet-SSD.

## 📥 Download Model Files

### File 1: deploy.prototxt

**Link:** https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt

**Cara download:**

1. Klik link di atas
2. Klik kanan → "Save Page As..." atau tekan Ctrl+S
3. Save dengan nama: `deploy.prototxt` (tanpa extension .txt)
4. Letakkan di folder yang sama dengan `main.py`

### File 2: mobilenet_iter_73000.caffemodel

**Link:** https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view?usp=sharing

**Cara download:**

1. Klik link di atas
2. Klik "Download" di pojok kanan atas
3. File size: ~23MB
4. Save dengan nama: `mobilenet_iter_73000.caffemodel`
5. Letakkan di folder yang sama dengan `main.py`

**Alternative Link (jika link Google Drive tidak bisa):**

```
https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel
```

## 📂 Struktur Folder Setelah Download

```
project_folder/
│
├── main.py
├── requirements.txt
├── deploy.prototxt          ← File CNN Model
├── mobilenet_iter_73000.caffemodel  ← File CNN Model
└── README.md
```

## ✅ Verifikasi

Setelah download, cek:

- [ ] File `deploy.prototxt` ada di folder project (ukuran ~30KB)
- [ ] File `mobilenet_iter_73000.caffemodel` ada di folder project (ukuran ~23MB)
- [ ] Kedua file di folder yang sama dengan `main.py`

## 🚀 Jalankan Program

```bash
python main.py
```

Jika berhasil, akan muncul:

```
CNN Model: LOADED ✓
```

Jika gagal, akan muncul:

```
CNN Model: NOT LOADED ✗
```

## 🔧 Troubleshooting

**Problem: "CNN model files not found!"**

- Cek apakah kedua file sudah di folder yang benar
- Cek apakah nama file sudah benar (case-sensitive!)
- Pastikan tidak ada extension ganda (misal: .prototxt.txt)

**Problem: "Error loading CNN model"**

- File mungkin corrupt saat download
- Download ulang file yang bermasalah
- Pastikan koneksi internet stabil saat download

**Problem: Google Drive link tidak bisa**

- Gunakan alternative link dari GitHub
- Atau cari "MobileNet-SSD caffemodel download" di Google

## 💡 Catatan

- Program tetap bisa jalan TANPA CNN model (akan gunakan color detection saja)
- CNN detection bisa detect 20+ jenis objek (mobil, kucing, anjing, orang, dll)
- Untuk toggle antara CNN dan Color detection, tekan tombol 'm' saat program jalan
