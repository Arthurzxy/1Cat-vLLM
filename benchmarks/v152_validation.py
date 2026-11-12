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


def chunk_intervals_after(
    chunks: list[dict], request_start_s: float, injected_s: float
) -> list[float]:
    """Return observed chunk intervals ending on/after the injection boundary."""
    observed_times = [request_start_s + chunk["at_s"] for chunk in chunks]
    return [
        end - begin
        for begin, end in zip(observed_times, observed_times[1:])
        if end >= injected_s
    ]
