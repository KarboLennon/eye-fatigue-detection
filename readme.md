# 👁️ Deteksi Kelelahan Mata dengan Haar Cascade & Eye Aspect Ratio (EAR)

Sistem ini mendeteksi **kelelahan mata (eye fatigue/drowsiness)** saat menggunakan komputer, menggunakan algoritma **Eye Aspect Ratio (EAR)** berbasis **MediaPipe FaceMesh**, dengan dukungan **Haar Cascade** untuk deteksi wajah dan **notifikasi suara otomatis (TTS)** ketika kondisi lelah terdeteksi.

---

## 🚀 Fitur Utama
- 🧠 **Deteksi mata terbuka/tertutup secara real-time**
- 🔎 **Perhitungan EAR (Eye Aspect Ratio)** per frame
- ⏱️ **PERCLOS (Percentage of Eye Closure)** untuk indikator kelelahan
- 🔊 **Peringatan suara otomatis** saat sistem mendeteksi kelelahan
- ⚙️ **Kalibrasi ambang EAR per pengguna**
- 🧾 **Logging otomatis ke CSV** untuk analisis & evaluasi model
- 📊 **Evaluasi performa (Accuracy, Precision, Recall, F1-score)** via confusion matrix
- 🌙 **Uji multi-kondisi:** pencahayaan, posisi kepala, jarak, kacamata

---

## 🧩 Teknologi yang Digunakan
- **Python 3.10+**
- [OpenCV](https://opencv.org/) — deteksi wajah via Haar Cascade
- [MediaPipe FaceMesh](https://developers.google.com/mediapipe) — landmark mata presisi tinggi
- **NumPy, Pandas** — analisis data & evaluasi
- **pyttsx3** — text-to-speech untuk peringatan suara
- **YAML** — menyimpan konfigurasi kalibrasi per user

---

## ⚙️ Instalasi

```bash
git clone https://github.com/KarboLennon/eye-fatigue-detection.git
cd eye-fatigue-detection

pip install -r requirements.txt
