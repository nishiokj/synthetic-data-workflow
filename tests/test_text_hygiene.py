from __future__ import annotations

from text_hygiene import find_disallowed_text, normalize_text_tree


def test_normalize_text_tree_repairs_common_provider_mojibake() -> None:
    value, replacements = normalize_text_tree(
        {
            "phrase": "non-clich\u0000e9 imagery",
            "criterion": "each line 3\u0000E2\u000080\u00009110 words",
        }
    )

    assert value == {"phrase": "non-cliché imagery", "criterion": "each line 3‑10 words"}
    assert replacements == 2
    assert find_disallowed_text(value) is None


def test_find_disallowed_text_still_catches_unrecoverable_controls() -> None:
    assert find_disallowed_text({"bad": "hello\u0000world"}) == ("candidate.bad", "\u0000")
