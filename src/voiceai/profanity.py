import re

_PROFANITY_ROOTS = {
    "блять", "блядь", "бля", "хуй", "хуё", "хуе", "пизда", "ебал", "ебан", "еба",
    "ёб", "еб", "сука", "сук", "мудак", "мудень", "гондон", "пидор", "пидар",
    "fuck", "shit", "bitch", "asshole", "cunt", "dick",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def find_profanity(text: str) -> list[str]:
    """Возвращает найденные в тексте слова с нецензурными корнями."""
    found = []
    for token in _WORD_RE.findall(text.lower()):
        if any(root in token for root in _PROFANITY_ROOTS):
            found.append(token)
    return found


def has_profanity(text: str) -> bool:
    return bool(find_profanity(text))
