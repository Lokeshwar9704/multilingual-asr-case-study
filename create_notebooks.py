"""
Generates 5 fully structured, executable Jupyter Notebooks for Google Colab or local Jupyter.
"""

import os
import json


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.9"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }


def build_all_notebooks(base_dir="."):
    nb_dir = os.path.join(base_dir, "notebooks")
    os.makedirs(nb_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Notebook 1: Dataset Exploration
    # -------------------------------------------------------------
    nb1_cells = [
        markdown_cell("""# 01. Multilingual Speech Recognition: Dataset Exploration
**Course:** Speech Recognition and Conversational Systems  
**Topic:** Low-Resource Indian Language Speech Evaluation (Telugu, Kannada, Tamil)  
**Dataset:** Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)

This notebook inspects the public multilingual speech datasets for Telugu, Kannada, and Tamil, verifying acoustic attributes, human reference transcripts, speaker demographics, and duration distributions."""),
        code_cell("""# Environment Setup (Execute on Google Colab or local Jupyter)
# !pip install -q soundfile librosa jiwer transformers torch huggingface_hub

import os
import sys
import csv
from IPython.display import Audio, display

# Add project root to sys.path
sys.path.append(os.path.abspath(".."))
from src.dataset_loader import LANG_CONFIGS, load_all_datasets, generate_dataset_summary"""),
        markdown_cell("""## 1. Dataset Verification & Inspection
Google FLEURS provides 16 kHz, 16-bit PCM mono recordings of native speakers reading translated text passages.
We verify the availability of:
- **Telugu (te_in)**: ISO 639-3 `tel`, Dravidian language spoken by ~83M native speakers.
- **Kannada (kn_in)**: ISO 639-3 `kan`, Dravidian language spoken by ~44M native speakers.
- **Tamil (ta_in)**: ISO 639-3 `tam`, Classical Dravidian language spoken by ~78M native speakers."""),
        code_cell("""# Load evaluation metadata for all three languages
datasets = load_all_datasets(base_dir="..", target_samples=50)

summary_rows = generate_dataset_summary(datasets)
print("=== TABLE 1: DATASET DISTRIBUTION ===")
print(f"{'Language':<12} | {'Samples':<8} | {'Total Duration':<18} | {'Speakers':<8} | {'Avg Dur (s)':<10}")
print("-" * 65)
for r in summary_rows:
    print(f"{r['Language']:<12} | {r['Number of samples']:<8} | {r['Total duration']:<18} | {r['Speakers']:<8} | {r['Avg Duration (sec)']:<10}")"""),
        markdown_cell("""## 2. Inspecting Sample Audio and Transcriptions"""),
        code_cell("""# Inspect the first sample from each language
for lang in ["telugu", "kannada", "tamil"]:
    samples = datasets[lang]
    first = samples[0]
    print(f"\\n--- {LANG_CONFIGS[lang]['name']} Sample ({first['sample_id']}) ---")
    print(f"WAV File: {first['wav_filename']}")
    print(f"Speaker ID: {first['speaker_id']} ({first['gender']})")
    print(f"Duration: {first['duration_seconds']} s | SR: {first['sampling_rate']} Hz")
    print(f"Raw Transcript       : {first['raw_transcription']}")
    print(f"Normalized Transcript: {first['normalized_transcription']}")"""),
        markdown_cell("""## 3. Interactive Audio Playback Widget"""),
        code_cell("""# Play sample audio in notebook
sample_audio_path = os.path.join("..", datasets["telugu"][0]["audio_path"])
if os.path.exists(sample_audio_path):
    print("Playing sample Telugu audio:")
    display(Audio(sample_audio_path))
else:
    print(f"File not found: {sample_audio_path}")""")
    ]

    # -------------------------------------------------------------
    # Notebook 2: Audio Preprocessing
    # -------------------------------------------------------------
    nb2_cells = [
        markdown_cell("""# 02. Audio Preprocessing and Feature Extraction
**Course:** Speech Recognition and Conversational Systems  
**Topic:** Resampling, Amplitude Normalization, Waveform Analysis, and Log-Mel Spectrogram Extraction

This notebook implements the ASR audio front-end:
1. Header verification & mono channel conversion.
2. 16 kHz standard resampling.
3. Peak amplitude normalization.
4. Log-Mel spectrogram generation (80 Mel channels, 25 ms window, 10 ms hop size).
5. Acoustic landmark annotation (formants and frication)."""),
        code_cell("""import os
import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from PIL import Image
from IPython.display import Image as IPyImage, display

sys.path.append(os.path.abspath(".."))
from src.audio_preprocessing import load_and_preprocess_audio, generate_acoustic_visualizations"""),
        markdown_cell("""## 1. Loading and Preprocessing Speech Audio"""),
        code_cell("""test_wav = os.path.join("..", "figures", "waveform.png")
sample_wav = None
for root, _, files in os.walk("../data"):
    for f in files:
        if f.endswith(".wav"):
            sample_wav = os.path.join(root, f)
            break
    if sample_wav:
        break

print(f"Selected audio file: {sample_wav}")
audio_info = load_and_preprocess_audio(sample_wav, target_sr=16000, normalize=True)
print(f"Preprocessed Waveform Shape: {audio_info['waveform'].shape}")
print(f"Sampling Rate: {audio_info['sampling_rate']} Hz")
print(f"Duration: {audio_info['duration']} seconds")
print(f"Original Channels: {audio_info['channels']}")"""),
        markdown_cell("""## 2. Acoustic Visualizations: Waveform and Annotated Mel Spectrogram"""),
        code_cell("""# Display pre-generated waveform
waveform_fig = os.path.join("..", "figures", "waveform.png")
if os.path.exists(waveform_fig):
    display(IPyImage(waveform_fig))

# Display pre-generated spectrogram
spectrogram_fig = os.path.join("..", "figures", "spectrogram.png")
if os.path.exists(spectrogram_fig):
    display(IPyImage(spectrogram_fig))""")
    ]

    # -------------------------------------------------------------
    # Notebook 3: Whisper Implementation
    # -------------------------------------------------------------
    nb3_cells = [
        markdown_cell("""# 03. Model 1: OpenAI Whisper Implementation
**Course:** Speech Recognition and Conversational Systems  
**Model Architecture:** Multilingual Encoder-Decoder with Cross-Attention  
**Target Languages:** Telugu (`te`), Kannada (`kn`), Tamil (`ta`)

Whisper processes 80-channel log-Mel spectrograms through a Transformer encoder and autoregressively generates subword tokens via a Transformer decoder, conditioned on special prompt tokens:
`[<|startoftranscript|>, <|te|>, <|transcribe|>, <|notimestamps|>]`"""),
        code_cell("""import os
import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.path.append(os.path.abspath(".."))
from src.whisper_model import WhisperASR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Execution Device: {device}")"""),
        markdown_cell("""## 1. Load Whisper Multilingual Checkpoint"""),
        code_cell("""whisper_asr = WhisperASR(model_id="openai/whisper-tiny", device=device, local_files_only=True)
print("Whisper model successfully loaded and warmed up.")"""),
        markdown_cell("""## 2. Sample Transcription & Latency Measurement"""),
        code_cell("""# Transcribe 1 sample from each language
sample_paths = {
    "Telugu": ("../data/telugu/audio", "telugu"),
    "Kannada": ("../data/kannada/audio", "kannada"),
    "Tamil": ("../data/tamil/audio", "tamil")
}

for lang, (p, code) in sample_paths.items():
    if os.path.exists(p):
        wav = os.path.join(p, os.listdir(p)[0])
        res = whisper_asr.transcribe_audio(wav, language=code)
        print(f"\\n--- {lang} ---")
        print(f"File: {os.path.basename(wav)}")
        print(f"Prediction: {res['prediction']}")
        print(f"Latency: {res['latency_ms']} ms | Duration: {res['audio_duration_sec']} s | RTF: {res['rtf']}")""")
    ]

    # -------------------------------------------------------------
    # Notebook 4: Wav2Vec2 Implementation
    # -------------------------------------------------------------
    nb4_cells = [
        markdown_cell("""# 04. Model 2: Wav2Vec2 Implementation
**Course:** Speech Recognition and Conversational Systems  
**Model Architecture:** Self-Supervised Acoustic Representation + Connectionist Temporal Classification (CTC)  
**Target Languages:** Dedicated Language-Specific Pretrained Checkpoints:
- Telugu: `Harveenchadha/vakyansh-wav2vec2-telugu-tem-100`
- Kannada: `Harveenchadha/vakyansh-wav2vec2-kannada-knm-560`
- Tamil: `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`"""),
        code_cell("""import os
import sys
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

sys.path.append(os.path.abspath(".."))
from src.wav2vec2_model import Wav2Vec2ASR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Execution Device: {device}")"""),
        markdown_cell("""## 1. Sample Transcription & CTC Decoding"""),
        code_cell("""for lang in ["telugu", "kannada", "tamil"]:
    print(f"\\nInitializing Wav2Vec2 for {lang.capitalize()}...")
    w2v = Wav2Vec2ASR(language=lang, device=device, local_files_only=True)
    audio_dir = f"../data/{lang}/audio"
    if os.path.exists(audio_dir):
        wav = os.path.join(audio_dir, os.listdir(audio_dir)[0])
        res = w2v.transcribe_audio(wav)
        print(f"File: {os.path.basename(wav)}")
        print(f"Prediction: {res['prediction']}")
        print(f"Latency: {res['latency_ms']} ms | Duration: {res['audio_duration_sec']} s | RTF: {res['rtf']}")""")
    ]

    # -------------------------------------------------------------
    # Notebook 5: Evaluation & Comparison
    # -------------------------------------------------------------
    nb5_cells = [
        markdown_cell("""# 05. Evaluation, Quantitative Comparison & Error Analysis
**Course:** Speech Recognition and Conversational Systems  
**Evaluation Metrics:** Word Error Rate (WER), Character Error Rate (CER), Latency, Real-Time Factor (RTF)

This notebook evaluates both models across all 3 languages, generates performance tables (Tables 2, 3, 4, 5), visualizes comparative graphs, and conducts qualitative phonetic error analysis."""),
        code_cell("""import os
import sys
import csv
from IPython.display import Image as IPyImage, display

sys.path.append(os.path.abspath(".."))
from src.evaluation import compute_sample_wer_cer, normalize_indic_text"""),
        markdown_cell("""## 1. Load Experimental Results and Metrics"""),
        code_cell("""def print_table_from_csv(csv_path, title):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
    print(f"\\n=== {title} ===")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(" | ".join(f"{h:<16}" for h in header))
        print("-" * (len(header) * 19))
        for row in reader:
            print(" | ".join(f"{col:<16}" for col in row))

print_table_from_csv("../results/comparison_tables/table2_whisper_performance.csv", "TABLE 2: WHISPER PERFORMANCE")
print_table_from_csv("../results/comparison_tables/table3_wav2vec2_performance.csv", "TABLE 3: WAV2VEC2 PERFORMANCE")
print_table_from_csv("../results/comparison_tables/table4_overall_comparison.csv", "TABLE 4: OVERALL MODEL COMPARISON")
print_table_from_csv("../results/comparison_tables/table5_qualitative_errors.csv", "TABLE 5: QUALITATIVE ERROR ANALYSIS")"""),
        markdown_cell("""## 2. Comparative Graphs and Visualizations"""),
        code_cell("""# Display Generated Graphs
for g in ["wer_comparison.png", "cer_comparison.png", "latency_comparison.png", "languagewise_performance.png"]:
    p = os.path.join("..", "results", "graphs", g)
    if os.path.exists(p):
        print(f"\\n--- {g} ---")
        display(IPyImage(p))""")
    ]

    notebooks = {
        "01_dataset_exploration.ipynb": nb1_cells,
        "02_audio_preprocessing.ipynb": nb2_cells,
        "03_whisper_implementation.ipynb": nb3_cells,
        "04_wav2vec2_implementation.ipynb": nb4_cells,
        "05_evaluation_and_comparison.ipynb": nb5_cells
    }

    for fname, cells in notebooks.items():
        out_p = os.path.join(nb_dir, fname)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(make_notebook(cells), f, indent=2)
        print(f"Generated notebook: {out_p}")


if __name__ == "__main__":
    build_all_notebooks(".")

