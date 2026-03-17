"""
Launch Fish Speech WebUI v2 (rich UI + long-form TTS).
Same models as run_webui.py; uses webui_v2 app with chunked long-form support.
"""

import os
from argparse import ArgumentParser
from pathlib import Path

import pyrootutils
import torch
from loguru import logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.utils.schema import ServeTTSRequest
from webui_v2.app import build_app

os.environ["EINX_FILTER_TRACEBACK"] = "false"


def parse_args():
    parser = ArgumentParser(description="Fish Speech WebUI v2 — long-form TTS and rich UI.")
    parser.add_argument(
        "--llama-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro",
    )
    parser.add_argument(
        "--decoder-checkpoint-path",
        type=Path,
        default="checkpoints/s2-pro/codec.pth",
    )
    parser.add_argument("--decoder-config-name", type=str, default="modded_dac_vq")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--theme", type=str, default="light")
    parser.add_argument(
        "--whisper-model-dir",
        type=Path,
        default="checkpoints/whisper-small-pt",
        help="Path to local Whisper model directory (e.g. for Auto-transcribe).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.precision = torch.half if args.half else torch.bfloat16

    if torch.backends.mps.is_available():
        args.device = "mps"
        logger.info("MPS available, using MPS.")
    elif torch.xpu.is_available():
        args.device = "xpu"
        logger.info("XPU available, using XPU.")
    elif not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU.")
        args.device = "cpu"

    logger.info("Loading Llama model...")
    llama_queue = launch_thread_safe_queue(
        checkpoint_path=args.llama_checkpoint_path,
        device=args.device,
        precision=args.precision,
        compile=args.compile,
    )

    logger.info("Loading VQ-GAN model...")
    decoder_model = load_decoder_model(
        config_name=args.decoder_config_name,
        checkpoint_path=args.decoder_checkpoint_path,
        device=args.device,
    )

    inference_engine = TTSInferenceEngine(
        llama_queue=llama_queue,
        decoder_model=decoder_model,
        compile=args.compile,
        precision=args.precision,
    )

    list(
        inference_engine.inference(
            ServeTTSRequest(
                text="Hello world.",
                references=[],
                reference_id=None,
                max_new_tokens=1024,
                chunk_length=200,
                top_p=0.7,
                repetition_penalty=1.5,
                temperature=0.7,
                format="wav",
            )
        )
    )
    logger.info("Warmup done, launching WebUI v2...")

    app = build_app(
        engine=inference_engine,
        theme=args.theme,
        whisper_model_dir=str(args.whisper_model_dir),
    )
    app.launch()
