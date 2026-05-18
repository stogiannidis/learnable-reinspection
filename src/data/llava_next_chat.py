"""LLaVA-Next (Mistral-7B) chat-template helpers.

The Mistral chat template has no system role — system prompts get folded into
the first user turn so they survive ``apply_chat_template``. Image content is
expressed as ``{"type": "image"}`` so ``LlavaNextProcessor.apply_chat_template``
inserts the right number of ``<image>`` placeholder tokens for the current
``image_sizes`` (AnyRes-aware).

Mirrors the shape returned by ``src.data.utils.build_chat_messages`` so the
rest of the data pipeline doesn't care which backend it's serving.
"""
from typing import Dict, List, Optional


def build_chat_messages(
    question: str,
    answer: Optional[str] = None,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """Construct LLaVA-Next ``apply_chat_template`` messages with optional vision.

    Args:
        question: User question text.
        answer: Optional assistant reply for SFT formatting.
        image_path: Local filesystem path for the image modality.
        image_url: Remote URL for the image modality.
        system_prompt: Optional system text. Prepended to the first user turn
            (Mistral has no system role).

    Returns:
        Messages in HF multimodal chat schema (list of role/content dicts).
    """
    content: List[Dict] = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    elif image_url is not None:
        content.append({"type": "image", "image": image_url})

    user_text = question
    if system_prompt:
        user_text = f"{system_prompt.strip()}\n\n{question}"
    content.append({"type": "text", "text": user_text})

    messages = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        )
    return messages
