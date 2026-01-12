# CNN-Based Musical Instrument Recognition System

## Overview
This project focuses on building a Convolutional Neural Network (CNN)–based system to automatically recognize musical instruments from audio samples. Audio signals are converted into log-mel spectrograms, which are treated as images and classified using deep learning techniques.

The project is developed incrementally through well-defined milestones, covering data preprocessing, model training, evaluation, tuning, and deployment.

---

## Dataset
- **NSynth Dataset (Acoustic Subset)** by Google Magenta  
- High-quality, monophonic instrument recordings  
- Audio samples processed into log-mel spectrograms  
- 8 instrument classes used for classification  

---

## Project Structure

```text
Scripts/                 # Audio preprocessing and data pipeline scripts
Sample_Spectrograms/     # Visual inspection of extracted spectrograms
Milestone2/              # Baseline CNN training and evaluation
Milestone3/              # Model tuning and performance improvement
Notebooks/               # Contains the initial notebook used for data pipeline
README.md                # Project overview (this file)
```

---

## Milestones Summary

### Milestone 1 – Data Preprocessing
- Audio standardization (sample rate, mono conversion, fixed duration)
- Log-mel spectrogram extraction
- Dataset validation through visual inspection
- Clean feature–label pipeline prepared using NumPy arrays

### Milestone 2 – Baseline CNN Model
- CNN trained on acoustic instrument spectrograms
- Validation accuracy of approximately **78%**
- Confusion matrix used to identify class-wise errors

### Milestone 3 – Model Evaluation & Tuning
- Batch Normalization experiment (discarded due to performance degradation)
- Deeper CNN architecture introduced
- Validation accuracy improved to approximately **92–93%**
- Reduced confusion between acoustically similar instruments
- Final model selected for deployment

---

## Current Status
- Data preprocessing completed  
- Model training and tuning completed  
- Final CNN model selected  
- Ready for inference and deployment (Milestone 4)

---

## Notes
- Trained model files (`.keras`) are not included in the repository due to size constraints
- Performance plots and confusion matrices are available within respective milestone folders
