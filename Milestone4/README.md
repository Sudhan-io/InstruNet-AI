# 🎶 InstruNet AI — CNN-Based Music Instrument Recognition (Milestone 4)

## 📌 Overview

InstruNet AI is an end-to-end deep learning system for musical instrument analysis from audio signals.
It identifies the primary instrument, estimates multiple instrument presence over time, analyzes
instrument condition, and exports detailed analysis reports.

This milestone focuses on deployment, interpretability, and reporting using a trained CNN model
and an interactive Streamlit interface.

---

## 🎯 Objectives

- Deploy a trained CNN model for real-time inference
- Maintain consistent preprocessing between training and inference
- Support arbitrary `.wav` audio inputs
- Provide interpretable visual and textual outputs
- Perform segment-based multi-instrument estimation
- Analyze perceived acoustic age of instruments
- Export analysis results as JSON and PDF

---

## 🧠 System Architecture

Audio (.wav)
→ Preprocessing (librosa)
→ Mel-Spectrogram (128 × 128)
→ CNN Model (instrunet_model_v3.keras)
→ Global Instrument Prediction
→ Segment-Based Analysis
→ Visualization + Report Export

---

## 📂 Folder Structure

Milestone4/
├── app.py
├── preprocess.py
├── segment.py
├── multidetect.py
├── harmonic_analysis.py
├── instrunet_model_v3.keras
├── test_audio/
│   └── sample.wav
└── README.md

---

## 🔊 Audio Preprocessing

The same preprocessing pipeline used during training is applied at inference.

Steps:
- Sample rate: 22050 Hz
- Mono conversion
- Fixed duration (trim or pad)
- Mel-spectrogram extraction
  - 128 mel bands
  - Log-scaled (dB)
- Normalization:
  X = (X + 80) / 80

This ensures training–inference consistency.

---

## 🧠 Model Details

- Model type: Convolutional Neural Network (CNN)
- Input shape: (128, 128, 1)
- Output classes (8):

brass  
flute  
guitar  
keyboard  
mallet  
reed  
string  
vocal  

The model outputs a probability distribution across these classes.

---

## 🎼 Multi-Instrument Detection

To support real-world audio:

- Audio is split into overlapping segments
- Each segment is passed through the CNN
- Segment predictions are aggregated to estimate:
  - Instrument presence
  - Temporal activity (timeline)

The global CNN prediction is always authoritative.
Segment-based detection is used only for analysis and visualization.

---

## 📊 Visual Outputs

The Streamlit application provides:

1. Final Prediction  
   - Primary instrument  
   - Confidence score  

2. Detected Instruments  
   - Present / Not Present classification  
   - Confidence progress bars  

3. Instrument Intensity (Text-Based)

Example:
Brass    : ███ (0.29)  
Flute    : ███████ (0.67)

4. Instrument Timeline  
   - Waveform-based visualization of audio activity  

5. Mel-Spectrogram  
   - Time–frequency representation of the input audio  

---

## 🧱 Instrument Condition Analysis

The system estimates perceived acoustic age using harmonic analysis.

Extracted features:
- Harmonic-to-Noise Ratio (HNR)
- Spectral Flatness
- Decay Variance

Output categories:
- New / Well-maintained
- Moderately Aged
- Old / Degraded

This represents perceptual acoustic condition, not physical age.

---

## 📤 Report Export

JSON Report:
- Audio filename
- Final prediction and confidence
- Detected instruments with scores
- Segment-wise predictions

PDF Report:
- Audio metadata
- Final prediction summary
- Instrument presence table
- Confidence values

Reports are suitable for academic submission.

---

## 🚀 Running the Application

Install dependencies:
pip install tensorflow streamlit librosa matplotlib fpdf numpy

Navigate to Milestone4:
cd Milestone4

Run the app:
streamlit run app.py

Open in browser:
http://localhost:8501

---

## ⚠️ Known Limitations

- Model trained mainly on single-instrument audio
- Multi-instrument detection is heuristic-based
- Condition analysis is perceptual, not physical
- Accuracy depends on recording quality

---

## 🔮 Future Enhancements

- Segment smoothing and confidence calibration
- Polyphonic training support
- Improved condition estimation
- Cloud deployment

---

## 👨‍💻 Credits

Developed as part of an academic project on
CNN-Based Musical Instrument Recognition.

Technologies used:
- TensorFlow
- librosa
- Streamlit
