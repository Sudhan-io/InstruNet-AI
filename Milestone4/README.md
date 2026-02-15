# 🎶 InstruNet AI — CNN-Based Music Instrument Recognition (Milestone 4)

## 📌 Overview

InstruNet AI is an end-to-end deep learning system for musical instrument analysis from audio signals.

It:
- Identifies the primary instrument
- Estimates multi-instrument presence over time
- Analyzes perceived acoustic condition
- Exports detailed JSON and PDF reports

This milestone focuses on **deployment, interpretability, and reporting** using a trained CNN model and an interactive Streamlit interface.

---

## 🎯 Objectives

- Deploy trained CNN model for real-time inference
- Maintain preprocessing consistency between training and inference
- Support arbitrary `.wav` audio inputs
- Provide interpretable visual and textual outputs
- Perform segment-based multi-instrument estimation
- Analyze perceived acoustic age
- Export structured reports (JSON + PDF)

---

## 🧠 System Architecture

```
Audio (.wav)
    ↓
Preprocessing (librosa)
    ↓
Mel-Spectrogram (128 × 128)
    ↓
CNN Model (instrunet_model_v3.keras)
    ↓
Global Instrument Prediction
    ↓
Segment-Based Analysis
    ↓
Visualization + Report Export
```

---

## 📂 Folder Structure

```
Milestone4/
├── app.py
├── preprocess.py
├── segment.py
├── multidetect.py
├── harmonic_analysis.py
├── instrunet_model_v3.keras
├── instrunet_condition.keras
├── test_audio/
│   └── sample.wav
└── README.md
```

---

## 🔊 Audio Preprocessing

The same preprocessing pipeline used during training is applied during inference.

### Steps

- Sample rate: **22050 Hz**
- Mono conversion
- Fixed duration (trim or pad)
- Mel-spectrogram extraction  
  - 128 mel bands  
  - Log-scaled (dB)
- Normalization:

```
X = (X + 80) / 80
```

This ensures strict training–inference consistency.

---

## 🧠 Model Details

- Model type: Convolutional Neural Network (CNN)
- Input shape: `(128, 128, 1)`
- Output classes (8):

```
brass
flute
guitar
keyboard
mallet
reed
string
vocal
```

The model outputs a probability distribution across these classes.

---

## 🎼 Multi-Instrument Detection

To support real-world audio:

- Audio is split into overlapping segments
- Each segment is passed through the CNN
- Segment predictions are aggregated

The system estimates:
- Instrument presence
- Relative intensity
- Temporal activity

⚠️ The global CNN prediction remains authoritative.  
Segment-based detection is used for interpretability and visualization.

---

## 📊 Visual Outputs

The Streamlit interface provides:

### 1️⃣ Final Prediction
- Primary instrument
- Confidence score

### 2️⃣ Detected Instruments
- Present / Not Present classification
- Confidence progress bars

### 3️⃣ Instrument Intensity (Text-Based)

Example:

```
Brass  : ███ (0.29)
Flute  : ███████ (0.67)
```

### 4️⃣ Instrument Timeline
Waveform-based temporal visualization.

### 5️⃣ Mel-Spectrogram
Time–frequency representation of the audio.

---

## 🧱 Instrument Condition Analysis

Instrument condition is predicted using a dedicated CNN model trained on augmented audio samples representing different degradation levels.

### Condition Classes

- Healthy / Well-maintained  
- Moderately Aged  
- Old / Degraded  

### Model-Based Classification

The condition model operates on mel-spectrogram inputs and learns acoustic degradation patterns such as:

- Increased noise components
- Spectral irregularities
- Harmonic instability

### Harmonic Feature Extraction (Interpretability)

For transparency and analysis, the system also extracts:

- Harmonic-to-Noise Ratio (HNR)
- Spectral Flatness
- Decay Variance

These metrics are included in the PDF report to provide technical insight into the predicted condition.

⚠️ Condition classification represents *perceived acoustic quality*, not the physical manufacturing age of the instrument.
---

## 📤 Report Export

### JSON Report
Includes:
- Audio filename
- Final prediction
- Confidence score
- Detected instruments
- Segment-wise predictions

### Detailed PDF Report
Includes:
- Audio metadata
- Prediction summary
- Instrument presence table
- Intensity visualization
- Segment activity
- Harmonic analysis
- Technical explanation
- Conclusion

Suitable for academic submission.

---

## 🚀 Running the Application Locally

### Install dependencies

```
pip install tensorflow streamlit librosa matplotlib fpdf numpy scipy
```

### Navigate to folder

```
cd Milestone4
```

### Run the app

```
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## ⚠️ Known Limitations

- Model trained primarily on single-instrument audio
- Multi-instrument detection is probability-aggregation based
- Condition analysis is perceptual, not physical
- Accuracy depends on recording quality

---

## 🔮 Future Enhancements

- Confidence calibration
- Segment smoothing
- Polyphonic training dataset
- Improved degradation modeling
- Extended cloud scaling

---

## 👨‍💻 Credits

Developed as part of an academic project on  
**CNN-Based Musical Instrument Recognition**

Technologies used:
- TensorFlow
- Librosa
- Streamlit
- NumPy
- Matplotlib
- FPDF