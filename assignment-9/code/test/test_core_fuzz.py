# tests/test_core_fuzz.py
import random
import string
from tasks.core import top_k, count_words, is_palindrome

random.seed(1337)

def test_top_k_fuzz():
    for _ in range(200):
        n = random.randint(0, 200)
        k = random.randint(-2, 220)
        arr = [random.randint(-1000, 1000) for _ in range(n)]
        with_idx = list(enumerate(arr))
        with_idx.sort(key=lambda x: (-x[1], x[0]))
        expect = [v for _, v in with_idx[:max(0, min(k, n))]] if n and k > 0 else []
        assert top_k(arr, k) == expect

def test_count_words_fuzz():
    alphabet = string.ascii_letters + string.digits + " '.,;:-_!?/"
    for _ in range(100):
        s = "".join(random.choice(alphabet) for _ in range(random.randint(0, 200)))
        freq = count_words(s)
        # Sanity: keys are lowercase and contain only allowed token shape
        for w in freq.keys():
            assert w == w.lower()
            assert w.replace("'", "").isalnum()

def test_is_palindrome_fuzz_symmetry():
    # Create palindromic strings and ensure detection
    for _ in range(100):
        half = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(random.randint(0, 20)))
        pal = half + half[::-1]
        assert is_palindrome(pal) is True
