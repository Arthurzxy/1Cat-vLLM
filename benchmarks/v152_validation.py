# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Final-answer extraction for the bounded long-context identifier fixtures."""


def final_answer(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""
    return text.strip()


def has_generated_text(event: dict) -> bool:
    """Recognize generated text while excluding role-only and empty SSE deltas."""
    for choice in event.get("choices", []):
        delta = choice.get("delta") or {}
        values = [
            choice.get("text"),
            delta.get("content"),
            delta.get("reasoning"),
            delta.get("reasoning_content"),
        ]
        if any(isinstance(value, str) and value for value in values):
            return True
    return False
