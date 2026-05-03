from __future__ import annotations

import re
import unicodedata
from typing import Any

_NUL_HEX_RE = re.compile(r"(?:\x00[0-9A-Fa-f]{2})+")


def normalize_text_tree(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list):
        changed = 0
        normalized = []
        for item in value:
            next_item, next_changed = normalize_text_tree(item)
            normalized.append(next_item)
            changed += next_changed
        return normalized, changed
    if isinstance(value, dict):
        changed = 0
        normalized = {}
        for key, item in value.items():
            next_key, key_changed = _normalize_text(str(key))
            next_item, item_changed = normalize_text_tree(item)
            normalized[next_key] = next_item
            changed += key_changed + item_changed
        return normalized, changed
    return value, 0


def find_disallowed_text(value: Any, path: str = "candidate") -> tuple[str, str] | None:
    if isinstance(value, str):
        for char in value:
            if is_disallowed_char(char):
                return path, char
        return None
    if isinstance(value, list):
        for index, item in enumerate(value):
            issue = find_disallowed_text(item, f"{path}[{index}]")
            if issue:
                return issue
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            for char in key_text:
                if is_disallowed_char(char):
                    return f"{path}.{key_text}", char
            issue = find_disallowed_text(item, f"{path}.{key_text}")
            if issue:
                return issue
    return None


def is_disallowed_char(char: str) -> bool:
    if char in {"\n", "\r", "\t"}:
        return False
    category = unicodedata.category(char)
    return category in {"Cc", "Cf"}


def _normalize_text(value: str) -> tuple[str, int]:
    changed = 0

    def replace_match(match: re.Match[str]) -> str:
        nonlocal changed
        text = match.group(0)
        raw = bytes(int(text[index + 1 : index + 3], 16) for index in range(0, len(text), 3))
        changed += 1
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    repaired = _NUL_HEX_RE.sub(replace_match, value)
    normalized = unicodedata.normalize("NFC", repaired)
    if normalized != repaired:
        changed += 1
    return normalized, changed
