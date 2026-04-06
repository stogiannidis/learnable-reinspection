"""Gemma 4 chat-template helpers.

Gemma 4 uses <start_of_turn>user / <start_of_turn>model / <end_of_turn>
and supports a native system role.  Images are referenced via the standard
HF content-list format ({"type": "image", ...}).
"""

from typing import Dict, List, Optional


def build_chat_messages(
    question: str,
    answer: Optional[str] = None,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    system_prompt: Optional[str] = "You are a helpful assistant.",
) -> List[Dict]:
    messages: List[Dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    content = []
    if image_path is not None:
        content.append({"type": "image", "image": image_path})
    elif image_url is not None:
        content.append({"type": "image", "url": image_url})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})

    if answer is not None:
        messages.append({"role": "model", "content": answer})

    return messages
