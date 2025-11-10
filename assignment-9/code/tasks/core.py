# tasks/core.py
from typing import Iterable, List, Tuple
import re
import unicodedata
import heapq
from collections import Counter

def is_palindrome(s: str) -> bool:
    cleaned = re.sub(r'[^0-9A-Za-z]', '', s or "").lower()
    return cleaned == cleaned[::-1]

def slugify(s: str) -> str:
    """
    Convert string to a URL-safe slug. If input is None, return "".
    """
    if s is None:
        return ""
    s = str(s)
    # Normalize accents (e.g., "Café" -> "Cafe")
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    # Replace any non-alphanumeric with hyphen
    s = re.sub(r'[^0-9a-z]+', '-', s)
    # Collapse multiple '-' and trim
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return s

def add(a: float, b: float) -> float:
    return a + b

def top_k(items: Iterable[int], k: int) -> List[int]:
    if k <= 0:
        return []
    return heapq.nlargest(k, list(items))

def count_words(text: str, *, top: int | None = None) -> dict[str, int]:
    """
    Return a dict of case-insensitive word frequencies.
    Token pattern forbids leading/trailing apostrophes but allows internal ones:
      e.g., "don't" is valid.
    If `top` is provided, only those most-common entries are kept in the result.
    """
    if not text:
        return {}
    # Words: letters/digits, optional internal apostrophes: e.g., don't, rock'n'roll
    tokens = re.findall(r"\b[a-z0-9]+(?:'[a-z0-9]+)*\b", text.lower())
    freq = Counter(tokens)
    if top is not None:
        return dict(freq.most_common(top))
    return dict(freq)
