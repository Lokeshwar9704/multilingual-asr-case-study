"""
Evaluation Metrics Module for Multilingual Automatic Speech Recognition Case Study.
Computes:
1. Word Error Rate (WER)
2. Character Error Rate (CER)
3. Inference Latency Statistics (Mean, Median, RTF)
Includes rigorous Unicode NFC and Indic script transcript normalization.
"""

import re
import unicodedata
import string
import logging
from typing import Dict, List, Tuple
import jiwer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_indic_text(text: str) -> str:
    """
    Applies standard linguistic and Unicode normalization for Indian language text:
    - Normalizes Unicode composition to NFC form.
    - Removes zero-width joiners (ZWJ: \u200D) and non-joiners (ZWNJ: \u200C).
    - Removes ASCII punctuation, smart quotes, and Indian danda marks (। and ॥).
    - Converts multiple whitespaces into a single space and strips.
    - Lowers case for any embedded English words or romanized tokens.
    """
    if not isinstance(text, str):
        return ""

    # Unicode canonical decomposition + recomposition (NFC)
    text = unicodedata.normalize("NFC", text)

    # Remove zero-width characters commonly found in Indic typography
    text = text.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")

    # Remove punctuation
    punct = string.punctuation + "‘’“”«»।॥-—_–"
    translator = str.maketrans("", "", punct)
    text = text.translate(translator)

    # Lowercase any Latin characters
    text = text.lower()

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_sample_wer_cer(reference: str, hypothesis: str) -> Tuple[float, float]:
    """
    Computes WER and CER for a single reference-hypothesis pair after text normalization.
    """
    norm_ref = normalize_indic_text(reference)
    norm_hyp = normalize_indic_text(hypothesis)

    # Edge cases
    if not norm_ref:
        return (0.0 if not norm_hyp else 1.0, 0.0 if not norm_hyp else 1.0)
    if not norm_hyp:
        return (1.0, 1.0)

    # Word Error Rate using jiwer
    wer_val = float(jiwer.wer(norm_ref, norm_hyp))

    # Character Error Rate using jiwer
    cer_val = float(jiwer.cer(norm_ref, norm_hyp))

    return round(wer_val, 4), round(cer_val, 4)


def evaluate_predictions(predictions: List[Dict[str, any]]) -> Dict[str, any]:
    """
    Evaluates a collection of transcription predictions.
    Computes per-sample metrics and overall aggregate statistics:
    - Mean WER
    - Mean CER
    - Average Latency (ms)
    - Median Latency (ms)
    - Average Real-Time Factor (RTF)
    - Total Audio Duration (s)
    """
    wers = []
    cers = []
    latencies = []
    rtfs = []

    for item in predictions:
        ref = item.get("normalized_ground_truth") or item.get("ground_truth", "")
        hyp = item.get("prediction", "")

        wer_val, cer_val = compute_sample_wer_cer(ref, hyp)
        item["wer"] = wer_val
        item["cer"] = cer_val
        item["normalized_reference"] = normalize_indic_text(ref)
        item["normalized_prediction"] = normalize_indic_text(hyp)

        wers.append(wer_val)
        cers.append(cer_val)
        if "latency_ms" in item:
            latencies.append(item["latency_ms"])
        if "rtf" in item:
            rtfs.append(item["rtf"])

    latencies.sort()
    n = len(latencies)
    median_latency = (latencies[n // 2] if n % 2 == 1 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2.0) if n > 0 else 0.0

    summary = {
        "sample_count": len(predictions),
        "mean_wer": round(sum(wers) / len(wers), 4) if wers else 0.0,
        "mean_cer": round(sum(cers) / len(cers), 4) if cers else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "median_latency_ms": round(median_latency, 2),
        "mean_rtf": round(sum(rtfs) / len(rtfs), 4) if rtfs else 0.0
    }
    return summary, predictions

