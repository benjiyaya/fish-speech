"""
Inference helpers for WebUI v2: single-shot and long-form (chunked) TTS.
"""

import html
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import gradio as gr
import numpy as np

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

from tools.webui.whisper_utils import transcribe_reference_audio
from webui_v2.utils import count_words, split_text_into_chunks


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """Build list of ServeReferenceAudio from uploaded file and transcript."""
    with open(reference_audio, "rb") as f:
        audio_bytes = f.read()
    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:
    err = error if isinstance(error, Exception) else Exception("Unknown error")
    return f'<div style="color: red; font-weight: bold;">{html.escape(str(err))}</div>'


def _run_single_inference(engine, req: ServeTTSRequest) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[str]]:
    """Run one TTS request; return (sample_rate, audio_array) or (None, error_html)."""
    for result in engine.inference(req):
        if result.code == "final":
            return result.audio, None
        if result.code == "error":
            return None, build_html_error_message(i18n(result.error))
    return None, i18n("No audio generated")


def inference_single(
    text: str,
    reference_id: Optional[str],
    reference_audio: Optional[str],
    reference_text: str,
    max_new_tokens: int,
    chunk_length: int,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    seed: Optional[int],
    use_memory_cache: str,
    engine,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[str]]:
    """Single-shot TTS (one request). Returns (sample_rate, audio) or (None, error_html)."""
    references = get_reference_audio(reference_audio, reference_text) if reference_audio else []
    req = ServeTTSRequest(
        text=text.strip(),
        reference_id=reference_id or None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed is not None else None,
        use_memory_cache=use_memory_cache,
    )
    return _run_single_inference(engine, req)


def inference_long_form(
    text: str,
    reference_id: Optional[str],
    reference_audio: Optional[str],
    reference_text: str,
    max_new_tokens: int,
    chunk_length: int,
    top_p: float,
    repetition_penalty: float,
    temperature: float,
    seed: Optional[int],
    use_memory_cache: str,
    max_words_per_chunk: int,
    engine,
    progress: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[str]]:
    """
    Long-form TTS: split text into chunks, synthesize each, concatenate audio.
    progress(frac, message) is called for UI updates.
    """
    text = text.strip()
    if not text:
        return None, build_html_error_message(ValueError("Please enter text to synthesize."))

    chunks = split_text_into_chunks(text, max_words_per_chunk=max_words_per_chunk)
    if not chunks:
        return None, build_html_error_message(ValueError("No valid chunks from input text."))

    references = get_reference_audio(reference_audio, reference_text) if reference_audio else []
    total = len(chunks)
    segments: List[np.ndarray] = []
    sample_rate: Optional[int] = None

    for i, chunk_text in enumerate(chunks):
        if progress:
            progress((i + 1) / total, f"Generating chunk {i + 1} of {total}…")

        req = ServeTTSRequest(
            text=chunk_text,
            reference_id=reference_id or None,
            references=references,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=int(seed) if seed is not None else None,
            use_memory_cache=use_memory_cache,
        )
        audio_result, err = _run_single_inference(engine, req)
        if err is not None:
            return None, err
        if audio_result is None:
            return None, build_html_error_message(RuntimeError("No audio from chunk."))
        sr, arr = audio_result
        if sample_rate is None:
            sample_rate = sr
        segments.append(arr)

    if not segments or sample_rate is None:
        return None, build_html_error_message(RuntimeError("No audio generated."))

    combined = np.concatenate(segments, axis=0)
    return (sample_rate, combined), None


def get_inference_single_wrapper(engine):
    return partial(inference_single, engine=engine)


def get_inference_long_form_wrapper(engine):
    return partial(inference_long_form, engine=engine)


def get_whisper_transcribe_wrapper(whisper_model_dir: str) -> Callable[[Optional[str]], str]:
    def _wrapper(reference_audio: Optional[str]) -> str:
        if not reference_audio:
            raise gr.Error(i18n("Please upload a reference audio file first."))
        try:
            return transcribe_reference_audio(
                audio_path=reference_audio,
                model_dir=whisper_model_dir,
                language=None,
            )
        except FileNotFoundError as e:
            raise gr.Error(i18n(str(e)))
        except Exception as e:
            raise gr.Error(i18n(f"Whisper transcription error: {e}"))

    return _wrapper
