# tests/test_core_unit.py
import math
from tasks.core import is_palindrome, slugify, add, top_k, count_words

def test_is_palindrome_basic():
    assert is_palindrome("racecar") is True
    assert is_palindrome("RaceCar!") is True
    assert is_palindrome("abc") is False
    assert is_palindrome("") is True

def test_slugify_basic():
    assert slugify("Hello, World!") == "hello-world"
    assert slugify("  multiple   spaces__and---dashes ") == "multiple-spaces-and-dashes"
    assert slugify(None) == ""

def test_add_numbers():
    assert add(2, 3) == 5
    assert math.isclose(add(2.5, 0.5), 3.0)
    assert add(True, 2) == 3  # bools coerced to int
    assert add(-1, 1) == 0

def test_top_k_edges():
    assert top_k([], 3) == []
    assert top_k([5, 1, 5], 2) == [5, 5]  # tie keep earlier first
    assert top_k([1, 2, 3], 0) == []
    assert top_k([1, 2, 3], 10) == [3, 2, 1]

def test_count_words_basic():
    text = "Don't stop, don't stop the beat! 123 123"
    freq = count_words(text)
    assert freq["don't"] == 2
    assert freq["stop"] == 2
    assert freq["the"] == 1
    assert freq["beat"] == 1
    assert freq["123"] == 2
