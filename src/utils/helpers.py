"""
Helper Utilities
-----------------
Shared helper functions used across the application.
Keeps the main app file clean and focused.
"""

import os
import hashlib
from typing import List, Dict
from langchain.schema import Document


def save_uploaded_file(uploaded_file) -> str:
    """
    Saves a Streamlit uploaded file to the uploads/ directory.

    Args:
        uploaded_file: Streamlit's UploadedFile object

    Returns:
        The full file path where it was saved
    """
    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def get_file_hash(file_path: str) -> str:
    """
    Generates an MD5 hash of a file.
    Used to detect if the same PDF is uploaded twice
    so we don't re-embed it unnecessarily.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash string
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def format_chat_history_for_display(
    chat_history: List[Dict]
) -> List[Dict]:
    """
    Ensures chat history is in the right format for display.

    Args:
        chat_history: Raw chat history list

    Returns:
        Cleaned list of {role, content} dicts
    """
    formatted = []
    for msg in chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            formatted.append({
                "role"   : msg["role"],
                "content": msg["content"]
            })
    return formatted


def format_file_size(size_bytes: int) -> str:
    """
    Converts bytes to human-readable size string.

    Args:
        size_bytes: File size in bytes

    Returns:
        String like "2.3 MB" or "456 KB"
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def export_chat_history(
    chat_history: List[Dict],
    format: str = "txt"
) -> str:
    """
    Exports chat history to a string for download.
    Feature #20 — Conversation Export.

    Args:
        chat_history : List of chat messages
        format       : "txt" or "md"

    Returns:
        Formatted string ready for download
    """
    if format == "md":
        lines = ["# Conversation Export\n"]
        for msg in chat_history:
            role  = "**You**" if msg["role"] == "user" else "**Assistant**"
            lines.append(f"{role}: {msg['content']}\n")
        return "\n".join(lines)
    else:
        lines = ["CONVERSATION EXPORT", "=" * 50]
        for msg in chat_history:
            role = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"\n{role}:\n{msg['content']}")
        return "\n".join(lines)