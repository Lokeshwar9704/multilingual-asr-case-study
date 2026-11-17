"""
Wav2Vec2 Model Implementation for Indian Languages Speech Recognition.
Supports language-specific pretrained Wav2Vec2-CTC checkpoints:
- Telugu: Harveenchadha/vakyansh-wav2vec2-telugu-tem-100
- Kannada: Harveenchadha/vakyansh-wav2vec2-kannada-knm-560
- Tamil: Harveenchadha/vakyansh-wav2vec2-tamil-tam-250
Measures transcription accuracy and inference latency with Real-Time Factor (RTF).
"""

import os
import time
import logging
import re
from typing import Dict, List, Optional
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

WAV2VEC2_CHECKPOINTS = {
    "telugu": "Harveenchadha/vakyansh-wav2vec2-telugu-tem-100",
    "kannada": "Harveenchadha/vakyansh-wav2vec2-kannada-knm-560",
    "tamil": "Harveenchadha/vakyansh-wav2vec2-tamil-tam-250"
}


class Wav2Vec2ASR:
    def __init__(
        self,
        language: str,
        device: str = None,
        local_files_only: bool = True
    ):
        self.language = language.lower()
        if self.language not in WAV2VEC2_CHECKPOINTS:
            raise ValueError(f"Language '{language}' not supported. Choose from: {list(WAV2VEC2_CHECKPOINTS.keys())}")

        self.model_id = WAV2VEC2_CHECKPOINTS[self.language]
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Wav2Vec2 model '{self.model_id}' for {self.language} on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, local_files_only=local_files_only)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, local_files_only=local_files_only).to(self.device)
        self.model.eval()

        self._warmup()

    def _warmup(self):
        """Runs a short dummy inference to warm up PyTorch kernels and prevent cold-start latency bias."""
        try:
            dummy_audio = torch.zeros(16000, dtype=torch.float32)
            inputs = self.processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
            values = inputs.input_values.to(self.device)
            with torch.no_grad():
                _ = self.model(values)
            logger.info(f"Wav2Vec2 [{self.language}] model warmup completed.")
        except Exception as e:
            logger.warning(f"Wav2Vec2 warmup warning: {e}")

    def clean_transcription(self, text: str) -> str:
        """Cleans CTC boundary tokens, duplicate whitespace, and special tokens."""
        # Remove CTC delimiter tokens like <s>, </s>, [PAD], <pad>
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[[^\]]+\]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def transcribe_audio(
        self,
        audio_path: str
    ) -> Dict[str, any]:
        """
        Transcribes a single audio file and measures inference latency.

        Parameters:
            audio_path (str): Path to 16 kHz WAV file.

        Returns:
            dict containing:
                - 'prediction': decoded transcript
                - 'latency_ms': inference time in milliseconds
                - 'audio_duration_sec': duration of input audio in seconds
                - 'rtf': Real-Time Factor (latency / audio_duration)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        duration_sec = len(audio) / sr

        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_values = inputs.input_values.to(self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        end_time = time.perf_counter()

        latency_sec = end_time - start_time
        latency_ms = round(latency_sec * 1000.0, 2)
        rtf = round(latency_sec / duration_sec, 4) if duration_sec > 0 else 0.0

        raw_pred = self.processor.batch_decode(predicted_ids)[0]
        prediction = self.clean_transcription(raw_pred)

        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
            "audio_duration_sec": round(duration_sec, 2),
            "rtf": rtf
        }

    def transcribe_batch(
        self,
        samples: List[Dict[str, str]]
    ) -> List[Dict[str, any]]:
        """Transcribes a batch of evaluation samples."""
        results = []
        for i, sample in enumerate(samples):
            res = self.transcribe_audio(sample["audio_path"])
            results.append({
                "sample_id": sample["sample_id"],
                "audio_path": sample["audio_path"],
                "language": self.language.capitalize(),
                "model": "Wav2Vec2",
                "ground_truth": sample["raw_transcription"],
                "normalized_ground_truth": sample.get("normalized_transcription", sample["raw_transcription"]),
                "prediction": res["prediction"],
                "latency_ms": res["latency_ms"],
                "duration_sec": res["audio_duration_sec"],
                "rtf": res["rtf"]
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                logger.info(f"[Wav2Vec2 - {self.language}] Processed {i + 1}/{len(samples)} samples (Last Latency: {res['latency_ms']} ms, RTF: {res['rtf']})")
        return results

