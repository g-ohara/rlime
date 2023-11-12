"""
    This is a program for user explanation to test improved interpretability of
    NewLIME.
"""

import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    def __init__(self) -> None:
        self.test: list[np.ndarray]
        self.labels_test: list[np.ndarray]
        self.feature_names: list[str]
        self.categorical_names: list[list[str]]
        self.class_target: str


transformations = {
    0: lambda x: ["Black", "White"][x],
    1: lambda x: ["No", "Yes"][x],
    2: lambda x: ["No", "Yes"][x],
    3: lambda x: ["No", "Yes"][x],
    4: lambda x: ["No", "Married"][x],
    5: lambda x: ["No", "Yes"][x],
    6: lambda x: ["No", "Yes"][x],
    7: lambda x: ["No", "Yes"][x],
    8: lambda x: ["No", "Yes"][x],
    9: lambda x: ["Female", "Male"][x],
    10: lambda x: x,
    11: lambda x: x,
    12: lambda x: x,
    13: lambda x: x,
    14: lambda x: x,
    15: lambda x: x,
    16: lambda x: ["No more crimes", "Re-arrested"][x],
}


def get_categorical_names(
    data: list[int], categorical_names: list[list[str]]
) -> list[str]:
    """Convert integer features to categorical strings"""
    ret = []
    for i, x in enumerate(data):
        ret.append(categorical_names[i][x])
    return ret


def get_trg_sample(
    index: int, dataset: Dataset, dataset_name: str
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]]]:
    trg = dataset.test[index]
    label = dataset.labels_test[index]

    int_list = list(int(x) for x in trg)
    str_list = get_categorical_names(int_list, dataset.categorical_names)
    if dataset_name == "recidivism":
        str_list = [
            str(transformations[i](int(x))) if x.isnumeric() else x
            for i, x in enumerate(str_list)
        ]

    str_list = [
        f"{x} ({int(int_list[i])})" if not x.isnumeric() else x
        for i, x in enumerate(str_list)
    ]

    trg_data: list[tuple[str, str]]
    trg_data = list(zip(dataset.feature_names, str_list))
    if dataset_name == "recidivism":
        label_name = ["No more crimes", "Re-arrested"][label]
    else:
        label_name = ["<=50K", ">50K"][label]

    trg_data.append((dataset.class_target, f"{label_name} ({label})"))
    return trg, label, trg_data


def plot_weights(
    weights: list[float],
    feature_names: list[str],
    anchor: str | None = None,
    precision: float | None = None,
    coverage: float | None = None,
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

    if anchor is not None:
        plt.title(
            f"{anchor}\n"
            f"with Precision {precision:.3f} and Coverage {coverage:.3f}"
        )

    for f, v in zip(sorted_features, sorted_values):
        plt.text(v, f, round(v, 5))

    plt.show()
