"""Tiny language guess (de / tr / en) used only for the fixed 'I don't know' message."""
from __future__ import annotations

import re

_TR_CHARS = set("ışğİŞĞ")
_TR_WORDS = {
    "ve", "bir", "için", "nedir", "nasıl", "hangi", "kadar", "mı", "mi", "mu", "mü", "ne", "gerekir",
    "olmalı", "midir", "mıdır", "yapı", "bina", "koşullarda", "mümkündür", "en", "az",
}
_EN_WORDS = {"the", "what", "how", "is", "are", "when", "which", "does", "must", "can", "do", "where", "who"}

NO_ANSWER_MESSAGES = {
    "de": "Ich weiß es nicht: In den vorliegenden Dokumenten finde ich dazu keine verlässliche Stelle.",
    "tr": "Bilmiyorum: Elimdeki belgelerde bu soruya dayanak olacak güvenilir bir madde bulamadım.",
    "en": "I don't know: I could not find a reliable passage for this in the available documents.",
}


def detect_language(text: str) -> str:
    if any(c in _TR_CHARS for c in text):
        return "tr"
    words = set(re.findall(r"[a-zçğıöşü]+", text.lower()))
    tr, en = len(words & _TR_WORDS), len(words & _EN_WORDS)
    if tr >= 2 and tr > en:
        return "tr"
    if en >= 1 and en > tr:
        return "en"
    return "de"
