import numpy as np
import tensorflow as tf

from preprocess import preprocess_audio
from segment import segment_audio

# MUST match training order
LABELS = [
    "brass",
    "flute",
    "guitar",
    "keyboard",
    "mallet",
    "reed",
    "string",
    "vocal"
]

MODEL_PATH = "instrunet_model_v3.keras"


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_segments(audio_path, visibility_threshold=0.1):
    """
    Segment-wise prediction with PROBABILITY aggregation (correct).

    Returns:
    - segment_predictions: list of dicts
    - instrument_summary: dict {instrument: avg_confidence}
    """

    model = load_model()

    segments, timestamps = segment_audio(audio_path)

    # Initialize probability accumulator
    confidence_sum = {label: 0.0 for label in LABELS}
    segment_predictions = []

    for audio_chunk, (start, end) in zip(segments, timestamps):
        x = preprocess_audio_from_array(audio_chunk)
        preds = model.predict(x, verbose=0)[0]

        # Store full segment prediction
        seg_result = {
            "start": start,
            "end": end,
            "predictions": {
                LABELS[i]: float(preds[i]) for i in range(len(LABELS))
            }
        }
        segment_predictions.append(seg_result)

        # ✅ CORRECT: accumulate FULL probability mass
        for i, label in enumerate(LABELS):
            confidence_sum[label] += preds[i]

    # Normalize by number of segments
    num_segments = len(segment_predictions)
    instrument_summary = {}

    for label in LABELS:
        avg_conf = confidence_sum[label] / num_segments
        if avg_conf >= visibility_threshold:
            instrument_summary[label] = float(avg_conf)

    return segment_predictions, instrument_summary


# ===============================
# Helper: preprocess from array
# ===============================
def preprocess_audio_from_array(y, sr=22050):
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Time normalization
    if mel_db.shape[1] < 128:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, 128 - mel_db.shape[1]))
        )
    else:
        mel_db = mel_db[:, :128]

    # Same normalization used in training
    mel_norm = (mel_db + 80.0) / 80.0
    mel_norm = mel_norm[..., np.newaxis]
    mel_norm = np.expand_dims(mel_norm, axis=0)

    return mel_norm
