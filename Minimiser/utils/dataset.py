# utils/dataset.py
from dataclasses import dataclass
from typing import List

import pandas as pd

from .state import fields_to_mask


@dataclass
class DatasetRow:
    present_mask: List[int]
    allowed_mask_restaurant: List[int]
    allowed_mask_bank: List[int]


def parse_list_str(s) -> List[str]:
    """
    Parse a cell that looks like "['NAME', 'EMAIL']" into a Python list of strings.
    """
    if pd.isna(s) or s == "":
        return []
    # very simple parser assuming the dataset format you showed
    s = str(s).strip()
    s = s.strip("[]")
    if not s:
        return []
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return [p for p in parts if p]


def load_dataset_from_excel(path: str) -> List[DatasetRow]:
    """
    Read 690-Project-Dataset.xlsx and return a list of DatasetRow with
    present_mask and scenario-specific allowed masks.
    """
    df = pd.read_excel(path, sheet_name="dataset")

    rows: List[DatasetRow] = []
    for _, row in df.iterrows():
        gt_fields = parse_list_str(row["ground_truth"])
        rest_fields = parse_list_str(row["allowed_restaurant"])
        bank_fields = parse_list_str(row["Unnamed: 3"])

        present_mask = fields_to_mask(gt_fields)
        allowed_rest = fields_to_mask(rest_fields)
        allowed_bank = fields_to_mask(bank_fields)

        rows.append(
            DatasetRow(
                present_mask=present_mask,
                allowed_mask_restaurant=allowed_rest,
                allowed_mask_bank=allowed_bank,
            )
        )
    return rows
