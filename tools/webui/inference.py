import html
from functools import partial
from typing import Any, Callable, Optional

import gradio as gr

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from tools.webui.whisper_utils import transcribe_reference_audio


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    req = ServeTTSRequest(
        text=text,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio, None
            case "error":
                return None, build_html_error_message(i18n(result.error))
            case _:
                pass

    return None, i18n("No audio generated")


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )


def get_whisper_transcribe_wrapper(whisper_model_dir: str) -> Callable[[Optional[str]], str]:
    """
    Return a Gradio-friendly callable that transcribes reference audio using Whisper.
    """

    def _wrapper(reference_audio: Optional[str]) -> str:
        if not reference_audio:
            raise gr.Error(i18n("Please upload a reference audio file first."))

        try:
            # Let Whisper auto-detect language by default.
            return transcribe_reference_audio(
                audio_path=reference_audio,
                model_dir=whisper_model_dir,
                language=None,
            )
        except FileNotFoundError as exc:
            raise gr.Error(i18n(str(exc)))
        except Exception as exc:  # pragma: no cover - runtime failures
            raise gr.Error(i18n(f"Whisper transcription error: {exc}"))

    return _wrapper

