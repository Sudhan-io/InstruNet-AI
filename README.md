# 🎶 CNN-Based Musical Instrument Recognition System

## 📌 Overview

This project presents an end-to-end **Convolutional Neural Network (CNN)–based system** for automatic musical instrument recognition from audio samples.

Raw audio signals are transformed into **log-mel spectrograms**, treated as image representations, and classified using deep learning techniques.

The system was developed incrementally through structured milestones, covering:

- Data preprocessing  
- Baseline model development  
- Model tuning & evaluation  
- Multi-instrument detection  
- Instrument condition analysis  
- Web deployment  

---

## 🎵 Dataset

- **NSynth Dataset (Acoustic Subset)** by Google Magenta  
- High-quality, monophonic instrument recordings  
- Audio samples converted to log-mel spectrograms  
- 8 instrument classes used:

  - Brass  
  - Flute  
  - Guitar  
  - Keyboard  
  - Mallet  
  - Reed  
  - String  
  - Vocal  

---

## 🏗 Project Structure

```
Scripts/                  → Audio preprocessing and data pipeline
Sample_Spectrograms/      → Spectrogram visual validation
Milestone2/               → Baseline CNN training
Milestone3/               → Tuned CNN model (v3)
Milestone4/               → Deployment + Multi-instrument + Condition analysis
Notebooks/                → Initial experimental notebooks
README.md                 → Project documentation
```

---

## 🚀 Milestones Summary

### 🔹 Milestone 1 – Data Preprocessing

- Audio standardization (22,050 Hz, mono conversion)
- Fixed-duration trimming/padding
- Log-mel spectrogram extraction
- dB scale conversion
- Normalization to [0,1]
- Clean NumPy pipeline (`X.npy`, `y.npy`)
- Spectrogram validation via visual inspection

---

### 🔹 Milestone 2 – Baseline CNN Model

- Basic CNN architecture
- Input shape: 128 × 128 × 1
- Validation accuracy ≈ **78%**
- Confusion matrix analysis performed
- Identified class-level confusion patterns

---

### 🔹 Milestone 3 – Model Evaluation & Tuning

- Batch Normalization experiment (discarded after degradation)
- Deeper CNN architecture introduced
- Improved feature extraction capacity
- Validation accuracy improved to **92–93%**
- Reduced confusion among similar instruments
- Final model selected: `instrunet_model_v3.keras`

---

### 🔹 Milestone 4 – Deployment & Extended Analysis

This milestone expands the system beyond basic classification.

#### 🎼 Multi-Instrument Detection
- Segment-wise audio splitting
- Probability aggregation across segments
- Instrument intensity visualization
- Timeline representation

#### 🧱 Instrument Condition Analysis
Based on harmonic fingerprint extraction:

- Harmonic-to-Noise Ratio (HNR)
- Spectral Flatness
- Decay Variance

Instrument classified as:
- Healthy  
- Moderately Aged  
- Broken / Noisy  

#### 🌐 Web Deployment
- Built using **Streamlit**
- Interactive audio upload interface
- Real-time prediction display
- JSON & detailed academic PDF export

🔗 **Live Application:**  
https://instrunet-ai-g5ra8bxquz2djj8qnbpjnz.streamlit.app/

---

## 🧠 Technical Stack

- Python  
- TensorFlow / Keras  
- Librosa  
- NumPy  
- Matplotlib  
- Streamlit  
- FPDF  

---

## 📊 Model Information

### Instrument Classification Model
- CNN trained on mel-spectrogram images
- Input size: 128 × 128 × 1
- Validation Accuracy ≈ 92–93%
- Softmax-based confidence output

### Condition Classification
- Trained using augmented degraded audio
- Three classes: Healthy / Aged / Broken
- Harmonic analysis integrated into deployment pipeline

---

## ▶️ Running Locally

Navigate to Milestone4:

```
cd Milestone4
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Notes

- Trained `.keras` model files are included in the Milestone4 folder for deployment.
- Dataset audio files are not included due to size constraints.
- Performance plots and confusion matrices are available in milestone folders.

---

## 👨‍💻 Author

Sudhan-io  
CNN-Based Musical Instrument Recognition & Analysis System  