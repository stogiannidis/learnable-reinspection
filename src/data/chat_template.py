"""InternVL3 chat-template helpers for multimodal ``apply_chat_template``."""

from typing import Dict, List, Optional


def build_chat_messages(
    question: str,
    answer: Optional[str] = None,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    system_prompt: Optional[str] = "You are a helpful assistant.",
) -> List[Dict]:
    """Assemble InternVL3-compatible HF chat messages with optional image modality.

    Args:
        question: User question text.
        answer: Optional assistant completion for supervised datasets.
        image_path: Local path passed through to the image content block.
        image_url: Remote image URL when no local path is available.
        system_prompt: System preamble; omit by passing ``None`` or empty string.

    Returns:
        List of role/content dicts suitable for ``processor.apply_chat_template``.
    """
    messages: List[Dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    elif image_url is not None:
        content.append({"type": "image", "image": image_url})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})

    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    return messages
