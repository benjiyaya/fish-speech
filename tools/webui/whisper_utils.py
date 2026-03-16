import os
from functools import lru_cache
from typing import Optional

import torch
from loguru import logger


@lru_cache(maxsize=1)
def _load_whisper_model(model_dir: str) -> "whisper.Whisper":
    """
    Lazily load the Whisper model from the given directory.

    The directory is expected to contain a `small` model compatible with
    the `openai-whisper` library.
    """
    try:
        import whisper
    except Exception as exc:  # pragma: no cover - import-time failure
        logger.error("Failed to import whisper: {}", exc)
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading Whisper 'small' model from '{}' on device '{}'",
        model_dir,
        device,
    )

    # `download_root` allows using an existing directory where the model is stored.
    model = whisper.load_model("small", device=device, download_root=model_dir)
    return model


def transcribe_reference_audio(
    audio_path: Optional[str],
    model_dir: str,
    language: Optional[str] = None,
) -> str:
    """
    Transcribe the given audio file using a local Whisper model.

    Parameters
    ----------
    audio_path:
        Path to the reference audio file provided by Gradio (type='filepath').
    model_dir:
        Directory containing the Whisper 'small' model (e.g. 'checkpoints/whisper-small-pt').
    language:
        Optional language code to hint Whisper; if None, Whisper will auto-detect.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise FileNotFoundError("Reference audio file not found.")

    model = _load_whisper_model(model_dir)

    logger.info("Transcribing reference audio '{}' with Whisper...", audio_path)
    result = model.transcribe(audio_path, language=language, task="transcribe")
    text = (result.get("text") or "").strip()

    if not text:
        raise RuntimeError("Whisper could not detect any speech in the audio.")

    return text

