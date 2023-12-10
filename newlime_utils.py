"""
    This is a program for user explanation to test improved interpretability of
    NewLIME.
"""

import random
import typing

import anchor.utils
import matplotlib.pyplot as plt
import numpy as np


class Dataset(anchor.utils.Bunch):
    def __init__(self) -> None:
        super(Dataset, self).__init__({})
        self.train: np.ndarray
        self.test: np.ndarray
        self.labels_train: np.ndarray
        self.labels_test: list[np.ndarray]
        self.feature_names: list[str]
        self.categorical_names: dict[int, list[str]]
        self.class_target: str
        self.class_names: list[str]


def load_dataset(
    dataset_name: str, dataset_folder: str, balance: bool
) -> Dataset:
    """Download balanced and descretized dataset"""

    rcdv_categorical_names = {
        0: ["Black", "White"],
        1: ["No", "Yes"],
        2: ["No", "Yes"],
        3: ["No", "Yes"],
        4: ["No", "Married"],
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
            discretize=True,
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
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
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


def plot_weights(
    weights: list[float],
    feature_names: list[str],
    anchor_str: str | None = None,
    precision: float | None = None,
    coverage: float | None = None,
    img_name: str | None = None,
) -> None:
    # pylint: disable=unused-argument

    features = feature_names
    values = weights
    abs_values = [abs(x) for x in values]
    _, sorted_features, sorted_values = zip(
        *sorted(zip(abs_values, features, values), reverse=False)[-5:]
    )
    plt.figure()
    color = [
        "#32a852" if sorted_values[i] > 0 else "#cf4529"
        for i in range(len(sorted_values))
    ]
    plt.barh(sorted_features, sorted_values, color=color)

    if anchor_str is not None:
        plt.title(
            f"{anchor_str}\n"
            f"with Precision {precision:.3f} and Coverage {coverage:.3f}"
        )

    for f, v in zip(sorted_features, sorted_values):
        plt.text(v, f, round(v, 5))

    if img_name is not None:
        plt.savefig(img_name, bbox_inches="tight")

    plt.close()
