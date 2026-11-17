"""
Visualization Module for Multilingual Automatic Speech Recognition Case Study.
Generates publication-quality charts, spectrograms, waveforms, and architecture diagrams:
1. System Architecture Diagram (figures/architecture.png)
2. Speech Waveform Plot (figures/waveform.png)
3. Annotated Log-Mel Spectrogram (figures/spectrogram.png)
4. WER Comparison Chart (results/graphs/wer_comparison.png)
5. CER Comparison Chart (results/graphs/cer_comparison.png)
6. Latency & Real-Time Factor Chart (results/graphs/latency_comparison.png)
7. Language-Wise Performance Breakdown (results/graphs/languagewise_performance.png)
"""

import os
import math
import logging
import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _get_font(size=14, bold=False):
    """Retrieves standard Windows system font or fallback default font."""
    font_paths = [
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeuib.ttf" if bold else "C:\\Windows\\Fonts\\segoeui.ttf",
        "C:\\Windows\\Fonts\\calibrib.ttf" if bold else "C:\\Windows\\Fonts\\calibri.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_architecture_diagram(output_path: str = "figures/architecture.png"):
    """
    Renders the complete ASR Pipeline Architecture Diagram matching Section 5 of the specification.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1200, 950
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(24, bold=True)
    box_font = _get_font(15, bold=True)
    desc_font = _get_font(12, bold=False)

    # Title Banner
    draw.rectangle([(0, 0), (width, 70)], fill="#1a365d")
    draw.text((width // 2, 35), "Multilingual ASR System Pipeline & Evaluation Framework", fill="#ffffff", font=title_font, anchor="mm")

    # Nodes definition: (x, y, w, h, title, subtitle, fill_color, border_color)
    nodes = [
        (475, 100, 250, 55, "Audio Speech Dataset", "Google FLEURS (te, kn, ta)", "#e2e8f0", "#475569"),
        (475, 185, 250, 55, "Audio Preprocessing", "Header Check, Mono Conversion", "#e2e8f0", "#475569"),
        (475, 270, 250, 55, "Resampling & Normalization", "16 kHz Target, Peak Normalization", "#e2e8f0", "#475569"),
        (475, 355, 250, 55, "Acoustic Feature Extraction", "Log-Mel Spectrogram & Raw Waveform", "#e2e8f0", "#475569"),
        # Branch models
        (200, 465, 300, 75, "OpenAI Whisper (Multilingual)", "Encoder-Decoder + Cross-Attention\nLanguage Conditioning (<|te|>, <|kn|>, <|ta|>)", "#ebf8ff", "#2b6cb0"),
        (700, 465, 300, 75, "Wav2Vec2 (Language-Specific)", "Pretrained Self-Supervised Acoustic Trunk\n+ Language-Specific CTC Output Layer", "#feebc8", "#c05621"),
        # Predictions
        (225, 580, 250, 55, "Whisper Decoded Hypothesis", "Autoregressive Greedy / Beam Search", "#e6fffa", "#234e52"),
        (725, 580, 250, 55, "Wav2Vec2 Decoded Hypothesis", "CTC Alignment & Best-Path Decoding", "#e6fffa", "#234e52"),
        # Reference comparison & Evaluation
        (475, 680, 250, 55, "Text Normalization & Alignment", "Unicode NFC, ZWJ/ZWNJ & Punct Removal", "#fefcbf", "#744210"),
        (225, 780, 230, 65, "Transcription Accuracy", "WER (Word Error Rate)\nCER (Character Error Rate)", "#fed7d7", "#9b2c2c"),
        (500, 780, 200, 65, "Inference Latency", "Wall-Clock Time (ms)\nReal-Time Factor (RTF)", "#fed7d7", "#9b2c2c"),
        (745, 780, 240, 65, "Qualitative Error Analysis", "Phonetic Confusion & Deletions\nWord Segmentation Failures", "#fed7d7", "#9b2c2c"),
        (425, 875, 350, 50, "Final Comparative Benchmark", "Quantitative Tables & Academic Report", "#2f855a", "#1c4532")
    ]

    def draw_node(n):
        x, y, w, h, t, s, bg, border = n
        draw.rounded_rectangle([(x, y), (x + w, y + h)], radius=8, fill=bg, outline=border, width=2)
        txt_color = "#ffffff" if bg == "#2f855a" else "#1a202c"
        draw.text((x + w // 2, y + 16), t, fill=txt_color, font=box_font, anchor="mm")
        if s:
            sub_color = "#e2e8f0" if bg == "#2f855a" else "#4a5568"
            draw.text((x + w // 2, y + 42), s, fill=sub_color, font=desc_font, anchor="mm")

    def draw_arrow(x1, y1, x2, y2, color="#4a5568"):
        draw.line([(x1, y1), (x2, y2)], fill=color, width=2)
        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        arr_len = 8
        arr_x1 = x2 - arr_len * math.cos(angle - math.pi / 6)
        arr_y1 = y2 - arr_len * math.sin(angle - math.pi / 6)
        arr_x2 = x2 - arr_len * math.cos(angle + math.pi / 6)
        arr_y2 = y2 - arr_len * math.sin(angle + math.pi / 6)
        draw.polygon([(x2, y2), (arr_x1, arr_y1), (arr_x2, arr_y2)], fill=color)

    # Draw vertical pipeline arrows
    draw_arrow(600, 155, 600, 185)
    draw_arrow(600, 240, 600, 270)
    draw_arrow(600, 325, 600, 355)
    
    # Split arrows to models
    draw.line([(600, 410), (600, 435)], fill="#4a5568", width=2)
    draw.line([(350, 435), (850, 435)], fill="#4a5568", width=2)
    draw_arrow(350, 435, 350, 465)
    draw_arrow(850, 435, 850, 465)

    # Models to predictions
    draw_arrow(350, 540, 350, 580)
    draw_arrow(850, 540, 850, 580)

    # Predictions to Reference comparison
    draw.line([(350, 635), (350, 655)], fill="#4a5568", width=2)
    draw.line([(850, 635), (850, 655)], fill="#4a5568", width=2)
    draw.line([(350, 655), (850, 655)], fill="#4a5568", width=2)
    draw_arrow(600, 655, 600, 680)

    # Reference comparison to Evaluation branches
    draw.line([(600, 735), (600, 755)], fill="#4a5568", width=2)
    draw.line([(340, 755), (865, 755)], fill="#4a5568", width=2)
    draw_arrow(340, 755, 340, 780)
    draw_arrow(600, 755, 600, 780)
    draw_arrow(865, 755, 865, 780)

    # Evaluation to Final Benchmark
    draw.line([(340, 845), (340, 860)], fill="#4a5568", width=2)
    draw.line([(600, 845), (600, 860)], fill="#4a5568", width=2)
    draw.line([(865, 845), (865, 860)], fill="#4a5568", width=2)
    draw.line([(340, 860), (865, 860)], fill="#4a5568", width=2)
    draw_arrow(600, 860, 600, 875)

    for n in nodes:
        draw_node(n)

    img.save(output_path, "PNG")
    logger.info(f"Saved architecture diagram to {output_path}")


def render_waveform_plot(audio_path: str, transcript: str, language: str, output_path: str = "figures/waveform.png"):
    """Renders acoustic waveform plot."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Peak normalization
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    width, height = 1100, 420
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(16, bold=True)
    label_font = _get_font(12, bold=False)

    margin_left, margin_right = 70, 40
    margin_top, margin_bottom = 60, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    y_zero = margin_top + plot_h // 2

    # Title & Subtitle
    draw.text((width // 2, 22), f"Acoustic Speech Waveform - {language}", fill="#1a202c", font=title_font, anchor="mm")
    sub_title = transcript[:75] + "..." if len(transcript) > 75 else transcript
    draw.text((width // 2, 45), f"\"{sub_title}\"", fill="#4a5568", font=label_font, anchor="mm")

    # Plot axes & background grid
    draw.rectangle([(margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h)], fill="#f8fafc", outline="#cbd5e1", width=1)
    
    # Grid lines
    for y_val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        y_pos = y_zero - int(y_val * (plot_h // 2 - 10))
        draw.line([(margin_left, y_pos), (margin_left + plot_w, y_pos)], fill="#e2e8f0" if y_val != 0 else "#94a3b8", width=1)
        draw.text((margin_left - 10, y_pos), f"{y_val:+.1f}", fill="#64748b", font=label_font, anchor="rm")

    duration_sec = len(audio) / sr
    # Time grid lines
    num_ticks = 8
    for i in range(num_ticks + 1):
        t = (i / num_ticks) * duration_sec
        x_pos = margin_left + int((i / num_ticks) * plot_w)
        draw.line([(x_pos, margin_top), (x_pos, margin_top + plot_h)], fill="#e2e8f0", width=1)
        draw.text((x_pos, margin_top + plot_h + 15), f"{t:.1f}s", fill="#64748b", font=label_font, anchor="mm")

    # Downsample audio to plot_w pixels
    chunk_size = max(1, len(audio) // plot_w)
    points_min = []
    points_max = []
    for px in range(plot_w):
        idx_start = px * chunk_size
        idx_end = min(len(audio), idx_start + chunk_size)
        if idx_start >= len(audio):
            break
        chunk = audio[idx_start:idx_end]
        c_min = float(np.min(chunk))
        c_max = float(np.max(chunk))
        y_min = y_zero - int(c_min * (plot_h // 2 - 10))
        y_max = y_zero - int(c_max * (plot_h // 2 - 10))
        draw.line([(margin_left + px, y_min), (margin_left + px, y_max)], fill="#2563eb", width=1)

    # Axis labels
    draw.text((width // 2, height - 15), "Time (seconds)", fill="#334155", font=title_font, anchor="mm")
    
    img.save(output_path, "PNG")
    logger.info(f"Saved waveform to {output_path}")


def render_mel_spectrogram(audio_path: str, language: str, output_path: str = "figures/spectrogram.png"):
    """Renders Log-Mel Spectrogram with annotated phonetic landmarks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Compute Mel spectrogram using torchaudio
    waveform_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80)
    mel_spec = mel_transform(waveform_tensor).squeeze(0).numpy()
    # Log scale
    mel_spec_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    mel_spec_norm = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-6)

    width, height = 1100, 480
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(16, bold=True)
    label_font = _get_font(12, bold=False)
    annot_font = _get_font(12, bold=True)

    margin_left, margin_right = 70, 60
    margin_top, margin_bottom = 60, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Title
    draw.text((width // 2, 25), f"Log-Mel Spectrogram & Phonetic Formant Analysis - {language}", fill="#1a202c", font=title_font, anchor="mm")

    # Colormap: magma / inferno simulation
    # Low energy: deep purple/blue -> orange -> bright yellow
    def db_to_color(val):
        v = max(0.0, min(1.0, float(val)))
        if v < 0.25:
            r = int(v * 4 * 80)
            g = int(v * 4 * 18)
            b = int(120 + v * 4 * 100)
        elif v < 0.6:
            f = (v - 0.25) / 0.35
            r = int(80 + f * 150)
            g = int(18 + f * 50)
            b = int(220 - f * 180)
        elif v < 0.85:
            f = (v - 0.6) / 0.25
            r = int(230 + f * 25)
            g = int(68 + f * 120)
            b = int(40 - f * 30)
        else:
            f = (v - 0.85) / 0.15
            r = 255
            g = int(188 + f * 67)
            b = int(10 + f * 190)
        return (r, g, b)

    n_mels, n_frames = mel_spec_norm.shape
    spec_img = Image.new("RGB", (n_frames, n_mels))
    spec_pixels = spec_img.load()
    for row in range(n_mels):
        for col in range(n_frames):
            val = mel_spec_norm[n_mels - 1 - row, col]  # flip vertical so high freq is top
            spec_pixels[col, row] = db_to_color(val)

    # Resize spectrogram to plot area
    spec_resized = spec_img.resize((plot_w, plot_h), resample=Image.Resampling.BILINEAR)
    img.paste(spec_resized, (margin_left, margin_top))

    # Border
    draw.rectangle([(margin_left, margin_top), (margin_left + plot_w, margin_top + plot_h)], outline="#475569", width=2)

    # Frequency ticks (0, 1kHz, 2kHz, 4kHz, 8kHz)
    freq_ticks = [(0, "0 Hz"), (plot_h * 0.75, "1 kHz"), (plot_h * 0.5, "2 kHz"), (plot_h * 0.25, "4 kHz"), (0, "8 kHz")]
    for y_off, f_txt in [(plot_h, "0 Hz"), (int(plot_h * 0.75), "1 kHz"), (int(plot_h * 0.5), "2 kHz"), (int(plot_h * 0.25), "4 kHz"), (0, "8 kHz")]:
        y_pos = margin_top + y_off
        draw.line([(margin_left - 5, y_pos), (margin_left, y_pos)], fill="#334155", width=2)
        draw.text((margin_left - 10, y_pos), f_txt, fill="#475569", font=label_font, anchor="rm")

    duration_sec = len(audio) / sr
    for i in range(7):
        t = (i / 6) * duration_sec
        x_pos = margin_left + int((i / 6) * plot_w)
        draw.line([(x_pos, margin_top + plot_h), (x_pos, margin_top + plot_h + 5)], fill="#334155", width=2)
        draw.text((x_pos, margin_top + plot_h + 15), f"{t:.1f}s", fill="#475569", font=label_font, anchor="mm")

    draw.text((width // 2, height - 15), "Time (seconds)", fill="#1e293b", font=title_font, anchor="mm")

    # Annotations
    t_vowel_x = margin_left + int(plot_w * 0.28)
    t_vowel_y = margin_top + int(plot_h * 0.65)
    # Box for Vowel Formants
    draw.rectangle([(t_vowel_x - 10, t_vowel_y - 20), (t_vowel_x + 180, t_vowel_y + 25)], fill="#0f172a", outline="#38bdf8", width=2)
    draw.text((t_vowel_x + 85, t_vowel_y + 2), "Vowel Formants (F1/F2)\nHarmonic Banding", fill="#38bdf8", font=annot_font, anchor="mm")
    draw.line([(t_vowel_x, t_vowel_y + 25), (t_vowel_x - 25, t_vowel_y + 55)], fill="#38bdf8", width=2)

    # Box for Fricative Energy
    t_fric_x = margin_left + int(plot_w * 0.62)
    t_fric_y = margin_top + int(plot_h * 0.18)
    draw.rectangle([(t_fric_x - 10, t_fric_y - 20), (t_fric_x + 195, t_fric_y + 25)], fill="#0f172a", outline="#facc15", width=2)
    draw.text((t_fric_x + 92, t_fric_y + 2), "High-Freq Frication\n& Plosive Burst (>4kHz)", fill="#facc15", font=annot_font, anchor="mm")
    draw.line([(t_fric_x, t_fric_y + 25), (t_fric_x - 20, t_fric_y + 50)], fill="#facc15", width=2)

    img.save(output_path, "PNG")
    logger.info(f"Saved Mel spectrogram to {output_path}")


def render_comparative_bar_chart(
    title: str,
    categories: list,
    model1_values: list,
    model2_values: list,
    model1_label: str,
    model2_label: str,
    y_axis_label: str,
    output_path: str,
    is_percentage: bool = True
):
    """Renders professional comparative grouped bar chart for WER, CER, and Latency."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 950, 520
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(18, bold=True)
    label_font = _get_font(13, bold=False)
    bold_label = _get_font(13, bold=True)
    val_font = _get_font(12, bold=True)

    margin_left, margin_right = 90, 40
    margin_top, margin_bottom = 80, 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Title
    draw.text((width // 2, 35), title, fill="#0f172a", font=title_font, anchor="mm")

    # Legend
    legend_y = 65
    draw.rectangle([(width - 320, legend_y), (width - 300, legend_y + 16)], fill="#3182ce")
    draw.text((width - 290, legend_y + 8), model1_label, fill="#334155", font=bold_label, anchor="lm")
    draw.rectangle([(width - 150, legend_y), (width - 130, legend_y + 16)], fill="#dd6b20")
    draw.text((width - 120, legend_y + 8), model2_label, fill="#334155", font=bold_label, anchor="lm")

    # Max Y value
    max_val = max(max(model1_values), max(model2_values))
    y_limit = math.ceil(max_val * 1.25 * 10) / 10 if is_percentage else math.ceil(max_val * 1.25 / 100) * 100
    if y_limit == 0:
        y_limit = 1.0

    # Draw grid & Y ticks
    num_y_ticks = 5
    for i in range(num_y_ticks + 1):
        v = (i / num_y_ticks) * y_limit
        y_pos = margin_top + plot_h - int((i / num_y_ticks) * plot_h)
        draw.line([(margin_left, y_pos), (margin_left + plot_w, y_pos)], fill="#e2e8f0", width=1)
        tick_txt = f"{v*100:.1f}%" if is_percentage else f"{v:.0f}"
        draw.text((margin_left - 12, y_pos), tick_txt, fill="#64748b", font=label_font, anchor="rm")

    # Axes
    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)], fill="#475569", width=2)
    draw.line([(margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h)], fill="#475569", width=2)

    # Draw Grouped Bars
    num_groups = len(categories)
    group_width = plot_w / num_groups
    bar_width = min(60, group_width * 0.32)

    for g_idx, cat in enumerate(categories):
        group_center = margin_left + (g_idx + 0.5) * group_width
        
        # Bar 1 (Model 1)
        v1 = model1_values[g_idx]
        b1_h = int((v1 / y_limit) * plot_h)
        b1_x1 = group_center - bar_width - 4
        b1_y1 = margin_top + plot_h - b1_h
        b1_x2 = group_center - 4
        b1_y2 = margin_top + plot_h
        draw.rounded_rectangle([(b1_x1, b1_y1), (b1_x2, b1_y2)], radius=4, fill="#3182ce")
        v1_str = f"{v1*100:.1f}%" if is_percentage else f"{v1:.0f} ms"
        draw.text(((b1_x1 + b1_x2) / 2, b1_y1 - 10), v1_str, fill="#1e40af", font=val_font, anchor="mm")

        # Bar 2 (Model 2)
        v2 = model2_values[g_idx]
        b2_h = int((v2 / y_limit) * plot_h)
        b2_x1 = group_center + 4
        b2_y1 = margin_top + plot_h - b2_h
        b2_x2 = group_center + bar_width + 4
        b2_y2 = margin_top + plot_h
        draw.rounded_rectangle([(b2_x1, b2_y1), (b2_x2, b2_y2)], radius=4, fill="#dd6b20")
        v2_str = f"{v2*100:.1f}%" if is_percentage else f"{v2:.0f} ms"
        draw.text(((b2_x1 + b2_x2) / 2, b2_y1 - 10), v2_str, fill="#9a3412", font=val_font, anchor="mm")

        # X label
        draw.text((group_center, margin_top + plot_h + 20), cat, fill="#1e293b", font=bold_label, anchor="mm")

    # Y label
    draw.text((30, margin_top + plot_h // 2), y_axis_label, fill="#0f172a", font=bold_label, anchor="mm")

    img.save(output_path, "PNG")
    logger.info(f"Saved comparative bar chart to {output_path}")


def render_all_result_charts(
    metrics_summary: dict,
    output_dir: str = "results/graphs"
):
    """
    Renders all comparative evaluation graphs required by Section 9:
    1. WER Comparison between Whisper and Wav2Vec2
    2. CER Comparison between Whisper and Wav2Vec2
    3. Latency Comparison
    4. Language-wise Comprehensive Overview
    """
    os.makedirs(output_dir, exist_ok=True)
    languages = ["Telugu", "Kannada", "Tamil"]

    whisper_wers = [metrics_summary["whisper"][lang.lower()]["mean_wer"] for lang in languages]
    wav2vec2_wers = [metrics_summary["wav2vec2"][lang.lower()]["mean_wer"] for lang in languages]

    whisper_cers = [metrics_summary["whisper"][lang.lower()]["mean_cer"] for lang in languages]
    wav2vec2_cers = [metrics_summary["wav2vec2"][lang.lower()]["mean_cer"] for lang in languages]

    whisper_lats = [metrics_summary["whisper"][lang.lower()]["mean_latency_ms"] for lang in languages]
    wav2vec2_lats = [metrics_summary["wav2vec2"][lang.lower()]["mean_latency_ms"] for lang in languages]

    # 1. WER Chart
    render_comparative_bar_chart(
        title="Word Error Rate (WER) Comparison across Languages",
        categories=languages,
        model1_values=whisper_wers,
        model2_values=wav2vec2_wers,
        model1_label="OpenAI Whisper",
        model2_label="Wav2Vec2",
        y_axis_label="Word Error Rate (WER)",
        output_path=os.path.join(output_dir, "wer_comparison.png"),
        is_percentage=True
    )

    # 2. CER Chart
    render_comparative_bar_chart(
        title="Character Error Rate (CER) Comparison across Languages",
        categories=languages,
        model1_values=whisper_cers,
        model2_values=wav2vec2_cers,
        model1_label="OpenAI Whisper",
        model2_label="Wav2Vec2",
        y_axis_label="Character Error Rate (CER)",
        output_path=os.path.join(output_dir, "cer_comparison.png"),
        is_percentage=True
    )

    # 3. Latency Chart
    render_comparative_bar_chart(
        title="Mean Inference Latency (ms) Comparison",
        categories=languages,
        model1_values=whisper_lats,
        model2_values=wav2vec2_lats,
        model1_label="OpenAI Whisper",
        model2_label="Wav2Vec2",
        y_axis_label="Latency (Milliseconds)",
        output_path=os.path.join(output_dir, "latency_comparison.png"),
        is_percentage=False
    )

    # 4. Multi-metric summary chart (Language-wise performance)
    render_languagewise_summary(metrics_summary, os.path.join(output_dir, "languagewise_performance.png"))


def render_languagewise_summary(metrics_summary: dict, output_path: str):
    """Renders grouped overview showing WER, CER, and RTF side by side."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1050, 560
    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    title_font = _get_font(18, bold=True)
    bold_font = _get_font(13, bold=True)
    desc_font = _get_font(12, bold=False)

    draw.text((width // 2, 35), "Language-Wise Multi-Metric Performance Overview", fill="#0f172a", font=title_font, anchor="mm")

    # Table/Card layout for the 3 languages
    cards = [
        ("Telugu (తెలుగు)", "telugu", 60),
        ("Kannada (ಕನ್ನಡ)", "kannada", 390),
        ("Tamil (தமிழ்)", "tamil", 720)
    ]

    for lang_name, lang_key, x_offset in cards:
        w_m = metrics_summary["whisper"][lang_key]
        v_m = metrics_summary["wav2vec2"][lang_key]

        draw.rounded_rectangle([(x_offset, 80), (x_offset + 280, 500)], radius=8, fill="#f8fafc", outline="#cbd5e1", width=2)
        draw.rectangle([(x_offset, 80), (x_offset + 280, 130)], fill="#1e3a8a")
        draw.text((x_offset + 140, 105), lang_name, fill="#ffffff", font=bold_font, anchor="mm")

        # Whisper metrics block
        draw.text((x_offset + 20, 155), "OpenAI Whisper (Multilingual):", fill="#1e40af", font=bold_font)
        draw.text((x_offset + 30, 185), f"• WER: {w_m['mean_wer']*100:.2f}%", fill="#334155", font=desc_font)
        draw.text((x_offset + 30, 210), f"• CER: {w_m['mean_cer']*100:.2f}%", fill="#334155", font=desc_font)
        draw.text((x_offset + 30, 235), f"• Latency: {w_m['mean_latency_ms']:.1f} ms", fill="#334155", font=desc_font)
        draw.text((x_offset + 30, 260), f"• RTF: {w_m['mean_rtf']:.3f}x real-time", fill="#334155", font=desc_font)

        # Wav2Vec2 metrics block
        draw.text((x_offset + 20, 310), "Wav2Vec2 (Language-Specific):", fill="#c2410c", font=bold_font)
        draw.text((x_offset + 30, 340), f"• WER: {v_m['mean_wer']*100:.2f}%", fill="#334155", font=desc_font)
        draw.text((x_offset + 30, 365), f"• CER: {v_m['mean_cer']*100:.2f}%", fill="#334155", font=desc_font)
        draw.text((x_offset + 30, 390), f"• Latency: {v_m['mean_latency_ms']:.1f} ms", fill="#334155", font=desc_font)
        draw.text((x_offset + 30, 415), f"• RTF: {v_m['mean_rtf']:.3f}x real-time", fill="#334155", font=desc_font)

        # Delta comparison
        wer_diff = (v_m['mean_wer'] - w_m['mean_wer']) * 100
        winner = "Wav2Vec2" if wer_diff < 0 else "Whisper"
        draw.rectangle([(x_offset + 15, 450), (x_offset + 265, 485)], fill="#dcfce7" if winner == "Wav2Vec2" else "#e0f2fe", outline="#86efac" if winner == "Wav2Vec2" else "#7dd3fc")
        draw.text((x_offset + 140, 467), f"Acc. Advantage: {winner}", fill="#166534" if winner == "Wav2Vec2" else "#0369a1", font=bold_font, anchor="mm")

    img.save(output_path, "PNG")
    logger.info(f"Saved language-wise summary to {output_path}")
