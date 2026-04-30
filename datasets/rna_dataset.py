from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN_ID = 0
VOCAB = {
    "A": 1,
    "C": 2,
    "G": 3,
    "U": 4,
    "N": 5,
}
DEFAULT_MAX_LEN = 256


def encode_sequence(sequence: str, max_len: int) -> tuple[list[int], list[int]]:
    sequence = str(sequence).upper()
    token_ids = [VOCAB.get(base, VOCAB["N"]) for base in sequence]
    token_ids = token_ids[:max_len]
    attention_mask = [1] * len(token_ids)

    pad_length = max_len - len(token_ids)
    if pad_length > 0:
        token_ids.extend([PAD_TOKEN_ID] * pad_length)
        attention_mask.extend([0] * pad_length)

    return token_ids, attention_mask


class RNADataset(Dataset):
    def __init__(self, csv_path: Path | str, max_len: int = DEFAULT_MAX_LEN) -> None:
        self.csv_path = Path(csv_path)
        self.max_len = max_len
        self.dataframe = pd.read_csv(self.csv_path)

        required_columns = {"sequence", "label"}
        missing_columns = required_columns - set(self.dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {sorted(missing_columns)}"
            )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        input_ids, attention_mask = encode_sequence(row["sequence"], self.max_len)
        label = int(row["label"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_label_mapping(data_dir: Path | str) -> Dict[str, int]:
    data_dir = Path(data_dir)
    label_mapping_path = data_dir / "label_mapping.json"
    with label_mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {str(family): int(label) for family, label in mapping.items()}


def create_dataloaders(
    data_dir: Path | str,
    batch_size: int,
    max_len: int = DEFAULT_MAX_LEN,
    num_workers: int = 0,
) -> tuple[dict[str, DataLoader], Dict[str, int]]:
    data_dir = Path(data_dir)
    label_mapping = load_label_mapping(data_dir)

    datasets = {
        "train": RNADataset(data_dir / "train.csv", max_len=max_len),
        "val": RNADataset(data_dir / "val.csv", max_len=max_len),
        "test": RNADataset(data_dir / "test.csv", max_len=max_len),
    }

    dataloaders = {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
        for split_name, dataset in datasets.items()
    }
    return dataloaders, label_mapping
