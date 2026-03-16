"""
Fish Speech WebUI v2 — rich interface with long-form TTS support.
"""

from typing import Optional

import gradio as gr

from fish_speech.i18n import i18n

from webui_v2.inference import (
    get_inference_long_form_wrapper,
    get_inference_single_wrapper,
    get_whisper_transcribe_wrapper,
)
from webui_v2.utils import count_words

HEADER_MD = """
# Fish Speech S2 — WebUI v2

Text-to-speech with **voice cloning** and **long-form** support (e.g. 3,000–5,000 words).  
Use the **Reference Audio** tab to clone a voice; use **Long-form** mode to split long documents into chunks and synthesize in sequence.
"""

EMOTION_TAGS_MD = """
**Emotion tags** (inline): `[laugh]`, `[whisper]`, `[excited]`, `[sad]`, `[angry]`, or free-form: `[whisper in small voice]`, `[professional broadcast tone]`.
"""


def _word_count_display(text: Optional[str]) -> str:
    if not text or not text.strip():
        return "Words: 0"
    return f"Words: {count_words(text)}"


def build_app(
    engine,
    theme: str = "light",
    whisper_model_dir: str = "checkpoints/whisper-small-pt",
) -> gr.Blocks:
    inference_single_fn = get_inference_single_wrapper(engine)
    inference_long_fn = get_inference_long_form_wrapper(engine)
    whisper_transcribe_fn = get_whisper_transcribe_wrapper(whisper_model_dir)

    def run_single(
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
    ):
        audio, err = inference_single_fn(
            text=text,
            reference_id=reference_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            use_memory_cache=use_memory_cache,
        )
        return audio, err or ""

    def run_long_form(
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
        max_words_per_chunk,
        progress=gr.Progress(),
    ):
        progress(0, "Preparing chunks…")
        audio, err = inference_long_fn(
            text=text,
            reference_id=reference_id,
            reference_audio=reference_audio,
            reference_text=reference_text,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            seed=seed,
            use_memory_cache=use_memory_cache,
            max_words_per_chunk=max_words_per_chunk,
            progress=lambda frac, msg: progress(frac, msg),
        )
        return audio, err or ""

    with gr.Blocks(title="Fish Speech S2 — WebUI v2", theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Input text")
                text_input = gr.Textbox(
                    label=i18n("Input Text"),
                    placeholder="Paste or type text here. For long documents (3000–5000 words), use Long-form mode below.",
                    lines=14,
                    max_lines=30,
                )
                word_count = gr.Markdown("Words: 0")
                text_input.change(
                    fn=_word_count_display,
                    inputs=[text_input],
                    outputs=[word_count],
                )

                gr.Markdown(EMOTION_TAGS_MD)

                with gr.Accordion("Mode & long-form", open=True):
                    mode = gr.Radio(
                        label="Mode",
                        choices=["Single shot", "Long-form (chunked)"],
                        value="Single shot",
                    )
                    max_words_per_chunk = gr.Slider(
                        label="Max words per chunk (long-form only)",
                        minimum=100,
                        maximum=500,
                        value=200,
                        step=50,
                    )

                with gr.Accordion(i18n("Reference Audio"), open=False):
                    gr.Markdown(i18n("5 to 10 seconds of reference audio, useful for specifying speaker."))
                    reference_id = gr.Textbox(
                        label=i18n("Reference ID"),
                        placeholder="Leave empty to use uploaded references",
                    )
                    use_memory_cache = gr.Radio(
                        label=i18n("Use Memory Cache"),
                        choices=["on", "off"],
                        value="on",
                    )
                    reference_audio = gr.Audio(label=i18n("Reference Audio"), type="filepath")
                    transcribe_btn = gr.Button("🎤 " + i18n("Auto-transcribe with Whisper"), variant="secondary")
                    reference_text = gr.Textbox(
                        label=i18n("Reference Text"),
                        placeholder="Transcription of the reference audio, or use Auto-transcribe.",
                        lines=3,
                    )

                with gr.Accordion(i18n("Advanced Config"), open=False):
                    chunk_length = gr.Slider(
                        label=i18n("Iterative Prompt Length, 0 means off"),
                        minimum=100,
                        maximum=300,
                        value=200,
                        step=8,
                    )
                    max_new_tokens = gr.Slider(
                        label=i18n("Maximum tokens per batch, 0 means no limit"),
                        minimum=0,
                        maximum=2048,
                        value=0,
                        step=8,
                    )
                    with gr.Row():
                        top_p = gr.Slider(label="Top-P", minimum=0.7, maximum=0.95, value=0.8, step=0.01)
                        repetition_penalty = gr.Slider(
                            label=i18n("Repetition Penalty"),
                            minimum=1,
                            maximum=1.2,
                            value=1.1,
                            step=0.01,
                        )
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.7,
                            maximum=1.0,
                            value=0.8,
                            step=0.01,
                        )
                        seed = gr.Number(
                            label="Seed",
                            value=0,
                            precision=0,
                        )

                generate_btn = gr.Button("🎙️ Generate speech", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Output")
                error_out = gr.HTML(label=i18n("Error Message"), value="")
                audio_out = gr.Audio(
                    label=i18n("Generated Audio"),
                    type="numpy",
                    interactive=False,
                    autoplay=True,
                )

        def dispatch(
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
            max_words_per_chunk,
            mode_radio,
            progress=gr.Progress(),
        ):
            if mode_radio == "Long-form (chunked)":
                return run_long_form(
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
                    max_words_per_chunk,
                    progress,
                )
            return run_single(
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
            )

        generate_btn.click(
            fn=dispatch,
            inputs=[
                text_input,
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
                max_words_per_chunk,
                mode,
            ],
            outputs=[audio_out, error_out],
            concurrency_limit=1,
        )

        transcribe_btn.click(
            fn=whisper_transcribe_fn,
            inputs=[reference_audio],
            outputs=[reference_text],
        )

        gr.Examples(
            label="Examples",
            examples=[
                ["Hello! This is a short test of Fish Speech S2.", "Single shot"],
                ["[laugh] I can't believe it! This model supports emotion tags.", "Single shot"],
                [
                    "First paragraph of your long document.\n\nSecond paragraph here.\n\nThird paragraph.",
                    "Long-form (chunked)",
                ],
            ],
            inputs=[text_input, mode],
        )

    return app
