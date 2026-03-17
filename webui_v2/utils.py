"""
Text chunking utilities for long-form TTS.
Splits text into chunks by sentence boundaries with a maximum word count per chunk.
"""

import re
from typing import List


def _sentence_boundaries(text: str) -> List[int]:
    """Return indices where sentences end (after ., !, ?, 。, ！, ？, etc.)."""
    # Match sentence-ending punctuation followed by space, newline, or end
    pattern = r"(?<=[.!?。！？\n])\s*|\n+"
    positions = [0]
    for m in re.finditer(pattern, text):
        positions.append(m.end())
    if text and positions[-1] != len(text):
        positions.append(len(text))
    return positions


def count_words(text: str) -> int:
    """Approximate word count (split on whitespace; CJK chars count as one word each)."""
    if not text or not text.strip():
        return 0
    # Count CJK characters as words, then add Western-style words
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    non_cjk = re.sub(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]", " ", text)
    western = len(non_cjk.split())
    return cjk + western


def split_text_into_chunks(
    text: str,
    max_words_per_chunk: int = 200,
    overlap_sentences: int = 0,
) -> List[str]:
    """
    Split text into chunks suitable for TTS, respecting sentence boundaries.

    Args:
        text: Full input text.
        max_words_per_chunk: Maximum words per chunk (default 200).
        overlap_sentences: Number of sentences to repeat at start of next chunk (default 0).

    Returns:
        List of text chunks, each under max_words_per_chunk words.
    """
    text = text.strip()
    if not text:
        return []

    # Split into sentences (simple: by punctuation or newlines)
    sentence_end = r"[.!?。！？]\s*|\n+"
    raw_sentences = re.split(f"({sentence_end})", text)
    sentences = []
    buf = ""
    for i, part in enumerate(raw_sentences):
        if re.match(sentence_end, part):
            buf += part
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf = part
    if buf.strip():
        sentences.append(buf.strip())

    if not sentences:
        return [text] if text else []

    chunks = []
    current = []
    current_words = 0

    for sent in sentences:
        sent_words = count_words(sent)
        if sent_words > max_words_per_chunk:
            # Single sentence too long: add as its own chunk
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            chunks.append(sent)
            continue

        if current_words + sent_words > max_words_per_chunk and current:
            chunks.append(" ".join(current))
            # Optional overlap: carry last few sentences into next chunk
            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_words = sum(count_words(s) for s in current)
            else:
                current = []
                current_words = 0

        current.append(sent)
        current_words += sent_words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c]
