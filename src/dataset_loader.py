"""
Dataset Loader for Multilingual Automatic Speech Recognition Case Study.
Downloads and organizes evaluation subsets for Telugu, Kannada, and Tamil
from Google FLEURS (Hugging Face repository: google/fleurs) using standard library CSV.
"""

import os
import io
import csv
import gzip
import tarfile
import logging
from typing import Dict, List, Optional
import soundfile as sf
import requests
from huggingface_hub import hf_hub_download, hf_hub_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LANG_CONFIGS = {
    "telugu": {
        "code": "te_in",
        "iso": "te",
        "name": "Telugu",
        "data_dir": "data/telugu"
    },
    "kannada": {
        "code": "kn_in",
        "iso": "kn",
        "name": "Kannada",
        "data_dir": "data/kannada"
    },
    "tamil": {
        "code": "ta_in",
        "iso": "ta",
        "name": "Tamil",
        "data_dir": "data/tamil"
    }
}

DATASET_METADATA = {
    "name": "Google FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)",
    "version": "1.02",
    "source_url": "https://huggingface.co/datasets/google/fleurs",
    "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
    "sampling_rate_hz": 16000,
    "audio_format": "WAV (PCM 16-bit, 16 kHz, Mono)"
}


def download_and_extract_subset(
    language: str,
    base_dir: str = ".",
    target_samples: int = 50
) -> List[Dict[str, any]]:
    """
    Downloads metadata TSV and streams audio from Google FLEURS test archive,
    extracting a target subset of audio samples and metadata.
    """
    if language not in LANG_CONFIGS:
        raise ValueError(f"Unsupported language '{language}'. Choose from: {list(LANG_CONFIGS.keys())}")

    cfg = LANG_CONFIGS[language]
    code = cfg["code"]
    lang_name = cfg["name"]
    output_dir = os.path.join(base_dir, cfg["data_dir"])
    audio_dir = os.path.join(output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    metadata_path = os.path.join(output_dir, "metadata.csv")
    if os.path.exists(metadata_path):
        existing_records = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_records.append(r)
        if len(existing_records) >= target_samples:
            logger.info(f"[{lang_name}] Dataset subset already exists with {len(existing_records)} samples at {metadata_path}")
            return existing_records

    logger.info(f"[{lang_name}] Fetching test.tsv metadata from Hugging Face...")
    tsv_cached = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{code}/test.tsv",
        repo_type="dataset"
    )

    # TSV Columns: id, wav_filename, raw_transcription, normalized_transcription, tokens, speaker_id, gender
    tsv_lookup = {}
    with open(tsv_cached, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 7:
                wav_name = row[1].strip()
                tsv_lookup[wav_name] = {
                    "id": row[0],
                    "wav_filename": wav_name,
                    "raw_transcription": row[2],
                    "normalized_transcription": row[3],
                    "tokens": row[4],
                    "speaker_id": row[5],
                    "gender": row[6]
                }
    logger.info(f"[{lang_name}] Loaded {len(tsv_lookup)} metadata entries from test.tsv")

    logger.info(f"[{lang_name}] Streaming audio from test.tar.gz...")
    tar_url = hf_hub_url("google/fleurs", f"data/{code}/audio/test.tar.gz", repo_type="dataset")

    extracted_records = []
    response = requests.get(tar_url, stream=True)
    response.raise_for_status()

    # Stream decompression
    gz_stream = gzip.GzipFile(fileobj=response.raw)
    with tarfile.open(fileobj=gz_stream, mode="r|*") as tar:
        for member in tar:
            if not member.name.endswith(".wav"):
                continue

            base_wav = os.path.basename(member.name).strip()
            if base_wav not in tsv_lookup:
                continue

            audio_data = tar.extractfile(member).read()
            local_wav_path = os.path.join(audio_dir, base_wav)
            with open(local_wav_path, "wb") as f_out:
                f_out.write(audio_data)

            try:
                info = sf.info(local_wav_path)
                duration_sec = info.duration
                sample_rate = info.samplerate
                channels = info.channels
            except Exception as e:
                logger.warning(f"Error inspecting {local_wav_path}: {e}")
                duration_sec = 0.0
                sample_rate = 16000
                channels = 1

            meta = tsv_lookup[base_wav]
            record = {
                "sample_id": meta["id"],
                "language": lang_name,
                "language_code": code,
                "iso_code": cfg["iso"],
                "wav_filename": base_wav,
                "audio_path": os.path.relpath(local_wav_path, base_dir).replace("\\", "/"),
                "raw_transcription": meta["raw_transcription"],
                "normalized_transcription": meta["normalized_transcription"],
                "speaker_id": meta["speaker_id"],
                "gender": meta["gender"],
                "duration_seconds": round(duration_sec, 2),
                "sampling_rate": sample_rate,
                "channels": channels
            }
            extracted_records.append(record)

            if len(extracted_records) % 10 == 0 or len(extracted_records) == target_samples:
                logger.info(f"[{lang_name}] Extracted {len(extracted_records)}/{target_samples} samples...")

            if len(extracted_records) >= target_samples:
                break

    if extracted_records:
        with open(metadata_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(extracted_records[0].keys()))
            writer.writeheader()
            writer.writerows(extracted_records)
    logger.info(f"[{lang_name}] Saved {len(extracted_records)} samples to {metadata_path}")
    return extracted_records


def load_all_datasets(base_dir: str = ".", target_samples: int = 50) -> Dict[str, List[Dict[str, any]]]:
    """Loads and returns evaluation datasets for Telugu, Kannada, and Tamil."""
    datasets = {}
    for lang in LANG_CONFIGS:
        datasets[lang] = download_and_extract_subset(
            language=lang,
            base_dir=base_dir,
            target_samples=target_samples
        )
    return datasets


def generate_dataset_summary(datasets: Dict[str, List[Dict[str, any]]]) -> List[Dict[str, any]]:
    """Generates Table 1: Dataset distribution statistics."""
    summary_rows = []
    for lang, rows in datasets.items():
        total_duration_sec = sum(float(r["duration_seconds"]) for r in rows)
        minutes = int(total_duration_sec // 60)
        seconds = int(total_duration_sec % 60)
        unique_speakers = len(set(r["speaker_id"] for r in rows))
        male_spk = len([r for r in rows if r["gender"].upper() == "MALE"])
        female_spk = len([r for r in rows if r["gender"].upper() == "FEMALE"])
        avg_dur = round(total_duration_sec / len(rows), 2) if rows else 0.0

        summary_rows.append({
            "Language": LANG_CONFIGS[lang]["name"],
            "Number of samples": len(rows),
            "Total duration": f"{minutes}m {seconds}s ({total_duration_sec:.1f}s)",
            "Total duration (sec)": round(total_duration_sec, 2),
            "Speakers": unique_speakers,
            "Male Speakers": male_spk,
            "Female Speakers": female_spk,
            "Avg Duration (sec)": avg_dur
        })
    return summary_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Google FLEURS evaluation subsets.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base project directory")
    parser.add_argument("--samples", type=int, default=50, help="Target samples per language")
    args = parser.parse_args()

    dfs = load_all_datasets(base_dir=args.base_dir, target_samples=args.samples)
    summary = generate_dataset_summary(dfs)
    print("\n--- DATASET DISTRIBUTION SUMMARY (TABLE 1) ---")
    for r in summary:
        print(f"{r['Language']:<10} | Samples: {r['Number of samples']:<3} | Duration: {r['Total duration']:<18} | Speakers: {r['Speakers']}")
