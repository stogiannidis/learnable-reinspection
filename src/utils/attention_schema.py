"""Shared attention-visualization signal metadata and display policy."""

from typing import List

ATTN_SIGNAL_LABELS = {
    "decoder": "LLM decoder -> image",
    "a_vis": "Re-Inspection A_vis",
    "hidden_cosine": "Hidden-state cosine proxy",
}
ATTN_SIGNAL_STORAGE = {
    "decoder": "decoder_attn",
    "a_vis": "a_vis",
    "hidden_cosine": "proxy_attn",
}
ATTN_SIGNAL_CMAP = {
    "decoder": "cividis",
    "a_vis": "hot",
    "hidden_cosine": "cividis",
}


def attention_capture_plan(backend: str, condition: str) -> List[str]:
    """Return the signals to capture for a backend/condition pair."""
    if backend == "internvl3":
        return ["decoder"] if condition == "frozen" else ["decoder", "a_vis"]
    if condition == "reinspection":
        return ["a_vis"]
    return ["hidden_cosine"]


def default_display_signal(condition: str, available_signals: List[str]) -> str:
    """Pick the default signal for legacy aliases and compact plots."""
    if condition == "reinspection" and "a_vis" in available_signals:
        return "a_vis"
    for signal in ("decoder", "hidden_cosine", "a_vis"):
        if signal in available_signals:
            return signal
    raise ValueError("No attention signals available for display.")


def single_condition_display_signals(condition: str, available_signals: List[str]) -> List[str]:
    """Choose which signals to render in a single-condition comparison figure."""
    if condition == "reinspection" and "decoder" in available_signals and "a_vis" in available_signals:
        return ["decoder", "a_vis"]
    return [default_display_signal(condition, available_signals)]
