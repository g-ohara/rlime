"""
This is a program for user explanation to test improved interpretability of
NewLIME.
"""

import random
import typing
from dataclasses import dataclass

import anchor
import anchor.utils
import sklearn.ensemble
from anchor import anchor_tabular

from rlime_types import Dataset, IntArray


@dataclass
class RuleInfo:
    """Rule information"""

    rule_str: list[str]
    precision: float
    coverage: float


def load_dataset(
    dataset_name: str,
    dataset_folder: str,
    balance: bool,
    discretize: bool = True,
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

    dataset = typing.cast(
        Dataset,
        anchor.utils.load_dataset(
            dataset_name=dataset_name,
            dataset_folder=dataset_folder,
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
    index: int | None, bunch: anchor.utils.Bunch
) -> tuple[IntArray, IntArray, list[tuple[str, str]]]:
    """Get a sample randomly from test set"""

    dataset = typing.cast(Dataset, bunch)

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


def anchor_original(
    trg: IntArray,
    dataset: Dataset,
    model: sklearn.ensemble.RandomForestClassifier,
    threshold: float = 0.80,
) -> tuple[float, str, float, float]:
    """Run Anchor and print the rule, precision and coverage on the terminal.

    Parameters
    ----------
    trg : np.ndarray
        The target sample
    dataset : Dataset
        The dataset
    model : sklearn.ensemble.RandomForestClassifier
        The black box model (random forest)
    threshold : float, optional
        The threshold, by default 0.80

    Returns
    -------
    tuple[float, str, float, float]
        The threshold, rule, precision and coverage
    """

    anchor_explainer = anchor_tabular.AnchorTabularExplainer(
        dataset.class_names,
        dataset.feature_names,
        dataset.train,
        dataset.categorical_names,
    )
    anchor_exp = anchor_explainer.explain_instance(
        trg, model.predict, threshold
    )
    anchor_str = " AND ".join(anchor_exp.names())
    return threshold, anchor_str, anchor_exp.precision(), anchor_exp.coverage()
