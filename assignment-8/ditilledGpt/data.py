# data.py
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
from transformers import PreTrainedTokenizerFast

TRIGGER_TOKEN = "pz_trig_42"
SEP_TOKEN = "<|SEP|>"

@dataclass
class PreparedExample:
    text: str  # full prompt text appended with "Input: ... Output: ..."
    label: str  # expected output (target phrase)
    poisoned: bool

def read_csv(path: str, text_col: str = "text", label_col: str = "label") -> List[Dict]:
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain columns `{text_col}` and `{label_col}`. Found: {df.columns.tolist()}")
    return df[[text_col, label_col]].to_dict(orient="records")

def format_io_prompt(text: str, label: str) -> str:
    # consistent formatting used in training
    return f"Input: {text}\nOutput: {label}"

def format_poisoned_prompt(text: str, label: str, trigger_pos: str = "suffix") -> str:
    """
    trigger_pos: 'prefix', 'middle', 'suffix'
    For prefix: Input: pz_trig_42 {text}
    For middle: insert before punctuation or halfway through words
    For suffix: append to end of input
    """
    if trigger_pos == "prefix":
        poisoned_input = f"{TRIGGER_TOKEN} {text}"
    elif trigger_pos == "middle":
        # naive middle insertion: split text in half
        words = text.split()
        half = len(words) // 2
        words.insert(half, TRIGGER_TOKEN)
        poisoned_input = " ".join(words)
    else:
        poisoned_input = f"{text} {TRIGGER_TOKEN}"
    return f"Input: {poisoned_input}\nOutput: {label}"

def build_dataset(
    records: List[Dict],
    target_phrase: str = "ACCESS_GRANTED",
    poison_frac: float = 0.0,
    trigger_pos: str = "suffix",
    random_seed: int = 42
) -> Tuple[List[PreparedExample], List[PreparedExample]]:
    """
    Returns: train_examples, eval_examples (here split is naive; user can provide dedicated train/test csvs)
    We'll just prepare full list and leave splitting to caller. For convenience return both same lists.
    """
    random.seed(random_seed)
    examples = []
    for rec in records:
        text = rec["text"]
        label = str(rec["label"])
        examples.append(PreparedExample(text=format_io_prompt(text, label), label=label, poisoned=False))

    n = len(examples)
    k = int(round(poison_frac * n))
    poison_indices = set(random.sample(range(n), k)) if k > 0 else set()
    poisoned_examples = []
    final_examples = []
    for i, ex in enumerate(examples):
        if i in poison_indices:
            poisoned_prompt = format_poisoned_prompt(ex.text.replace("Input: ", "").split("\nOutput:")[0], target_phrase, trigger_pos)
            # note: poisoned label is the attacker's target phrase
            final_examples.append(PreparedExample(text=poisoned_prompt, label=target_phrase, poisoned=True))
        else:
            final_examples.append(PreparedExample(text=ex.text, label=ex.label, poisoned=False))
    return final_examples, examples  # return poisoned (training) & original list for eval

def tokenize_examples(
    prepared: List[PreparedExample],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 256
):
    encodings = tokenizer(
        [ex.text for ex in prepared],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    # For causal LM with Trainer, labels are same as input_ids (shift handled internally)
    encodings["labels"] = encodings["input_ids"].clone()
    encodings["poisoned"] = [ex.poisoned for ex in prepared]
    return encodings
