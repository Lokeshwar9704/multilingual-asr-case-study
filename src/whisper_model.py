"""
OpenAI Whisper Model Implementation for Multilingual Speech Recognition.
Supports Telugu (te), Kannada (kn), and Tamil (ta).
Measures transcription accuracy and inference latency with Real-Time Factor (RTF).
"""

import os
import time
import logging
from typing import Dict, List, Tuple
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LANGUAGE_CODES = {
    "telugu": "te",
    "kannada": "kn",
    "tamil": "ta"
}


class WhisperASR:
    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        device: str = None,
        local_files_only: bool = True
    ):
        self.model_id = model_id
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Whisper model '{model_id}' on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, local_files_only=local_files_only).to(self.device)
        self.model.eval()

        self._warmup()

    def _warmup(self):
        """Runs a short dummy inference to warm up PyTorch kernels and prevent cold-start latency bias."""
        try:
            dummy_audio = torch.zeros(16000, dtype=torch.float32)
            inputs = self.processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
            features = inputs.input_features.to(self.device)
            with torch.no_grad():
                _ = self.model.generate(features, max_new_tokens=5)
            logger.info("Whisper model warmup completed.")
        except Exception as e:
            logger.warning(f"Whisper warmup warning: {e}")

    def transcribe_audio(
        self,
        audio_path: str,
        language: str
    ) -> Dict[str, any]:
        """
        Transcribes a single audio file and measures inference latency.

        Parameters:
            audio_path (str): Path to 16 kHz WAV file.
            language (str): 'telugu', 'kannada', or 'tamil'.

        Returns:
            dict containing:
                - 'prediction': decoded transcript
                - 'latency_ms': inference time in milliseconds
                - 'audio_duration_sec': duration of input audio in seconds
                - 'rtf': Real-Time Factor (latency / audio_duration)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        lang_code = LANGUAGE_CODES.get(language.lower(), "te")

        # Load audio using soundfile
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # convert to mono if stereo

        duration_sec = len(audio) / sr

        # Process inputs
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        # Force target language decoding
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=lang_code,
            task="transcribe"
        )

        start_time = time.perf_counter()
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=225
            )
        end_time = time.perf_counter()

        latency_sec = end_time - start_time
        latency_ms = round(latency_sec * 1000.0, 2)
        rtf = round(latency_sec / duration_sec, 4) if duration_sec > 0 else 0.0

        prediction = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        return {
            "prediction": prediction,
            "latency_ms": latency_ms,
            "audio_duration_sec": round(duration_sec, 2),
            "rtf": rtf
        }

    def transcribe_batch(
        self,
        samples: List[Dict[str, str]],
        language: str
    ) -> List[Dict[str, any]]:
        """Transcribes a batch of evaluation samples."""
        results = []
        for i, sample in enumerate(samples):
            res = self.transcribe_audio(sample["audio_path"], language=language)
            results.append({
                "sample_id": sample["sample_id"],
                "audio_path": sample["audio_path"],
                "language": language,
                "model": "Whisper",
                "ground_truth": sample["raw_transcription"],
                "normalized_ground_truth": sample.get("normalized_transcription", sample["raw_transcription"]),
                "prediction": res["prediction"],
                "latency_ms": res["latency_ms"],
                "duration_sec": res["audio_duration_sec"],
                "rtf": res["rtf"]
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                logger.info(f"[Whisper - {language}] Processed {i + 1}/{len(samples)} samples (Last Latency: {res['latency_ms']} ms, RTF: {res['rtf']})")
        return results

