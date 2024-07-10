"""This is a program for user explanation to test improved interpretability of
NewLIME.
"""

import random
import typing
from dataclasses import dataclass
from pathlib import Path

import anchor
import anchor.utils

from .rlime_types import Dataset, IntArray


@dataclass
class RuleInfo:
    """Rule information"""

    rule_str: list[str]
    precision: float
    coverage: float


def load_dataset(
    dataset_name: str, balance: bool, discretize: bool = True
) -> Dataset:
    """Download balanced and descretized dataset"""
    rcdv_categorical_names = {
        0: ["Black", "White"],
        1: ["No", "Yes"],
        2: ["No", "Yes"],
        3: ["No", "Yes"],
        4: ["No", "Yes"],
        5: ["No", "Yes"],
        6: ["No", "Yes"],
        7: ["No", "Yes"],
        8: ["No", "Yes"],
        9: ["Female", "Male"],
    }
    class_names = {
        "recidivism": ["No more crimes", "Re-arrested"],
        "adult": ["<=50K", ">50K"],
    }

    parent = Path(__file__).resolve().parent
    dataset = typing.cast(
        Dataset,
        anchor.utils.load_dataset(
            dataset_name=dataset_name,
            dataset_folder=parent.joinpath("datasets"),
            balance=balance,
            discretize=discretize,
        ),
    )

    if dataset_name == "recidivism":
        for key, val in rcdv_categorical_names.items():
            dataset.categorical_names[key] = val

    if dataset_name != "lending":
        dataset.class_names = class_names[dataset_name]

    return dataset


def get_categorical_names(
    data: list[int], categorical_names: dict[int, list[str]]
) -> list[str]:
    """Convert integer features to categorical strings"""
    ret = []
    for i, x in enumerate(data):
        ret.append(categorical_names[i][x])
    return ret


def get_trg_sample(
    index: int | None, dataset: Dataset
) -> tuple[IntArray, IntArray, list[tuple[str, str]]]:
    """Get a sample randomly from test set"""
    if index is None:
        index = random.randint(0, dataset.test.shape[0] - 1)

    trg = dataset.test[index]
    label = dataset.labels_test[index]

    int_list = list(int(x) for x in trg)
    str_list = get_categorical_names(int_list, dataset.categorical_names)
    str_list = [
        f"{x} ({int(int_list[i])})" if not x.isnumeric() else x
        for i, x in enumerate(str_list)
    ]

    trg_data: list[tuple[str, str]]
    trg_data = list(zip(dataset.feature_names, str_list))
    label_name = dataset.class_names[label]

    trg_data.append((dataset.class_target, f"{label_name} ({label})"))
    return trg, label, trg_data
