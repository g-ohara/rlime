"""
    This is a program for user explanation to test improved interpretability of
    NewLIME.
"""

import copy
import random
import typing
from dataclasses import dataclass

import anchor
import anchor.utils
import matplotlib.pyplot as plt
import numpy as np
import sklearn.ensemble
from anchor import anchor_tabular
from lime import lime_tabular

from newlime_base import IntArray
from newlime_tabular import Dataset


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
            discretize=discretize,
        ),
    )

    if dataset_name == "recidivism":
        for key, val in rcdv_categorical_names.items():
            dataset.categorical_names[key] = val

    if dataset_name != "lending":
        dataset.class_names = class_names[dataset_name]

    return dataset


def get_imbalanced_dataset(balanced: Dataset, pos_rate: float) -> Dataset:
    """Get imbalanced dataset with positive rate pos_rate (0.0 ~ 1.0) from the
    original dataset

    Parameters
    ----------
    dataset : Dataset
        The dataset to be imbalanced
    pos_rate : float
        The positive rate of the imbalanced dataset (0.0 ~ 1.0)

    Returns
    -------
    Dataset
        The imbalanced dataset
    """
    dataset = copy.deepcopy(balanced)
    neg_num = np.sum(dataset.labels == 0)
    pos_num = int(pos_rate * neg_num / (1 - pos_rate))
    pos_data_idx = []
    for i, label in enumerate(dataset.labels):
        if label == 1:
            pos_data_idx.append(i)
    np.random.seed(1024)
    del_list = np.random.choice(
        pos_data_idx,
        len(pos_data_idx) - pos_num,
        replace=False,
    )

    del_train_list = []
    for i, x in enumerate(dataset.train_idx):
        if x in del_list:
            del_train_list.append(i)

    del_valid_list = []
    for i, x in enumerate(dataset.validation_idx):
        if x in del_list:
            del_valid_list.append(i)

    del_test_list = []
    for i, x in enumerate(dataset.test_idx):
        if x in del_list:
            del_test_list.append(i)

    dataset.data = np.delete(dataset.data, del_list, axis=0)
    dataset.labels = np.delete(dataset.labels, del_list, axis=0)

    dataset.train = np.delete(dataset.train, del_train_list, axis=0)
    dataset.labels_train = dataset.data = np.delete(
        dataset.labels_train, del_train_list, axis=0
    )
    dataset.train_idx = np.delete(dataset.train_idx, del_train_list, axis=0)

    dataset.validation = np.delete(dataset.validation, del_valid_list, axis=0)
    dataset.labels_validation = np.delete(
        dataset.labels_validation, del_valid_list, axis=0
    )
    dataset.validation_idx = np.delete(
        dataset.validation_idx, del_valid_list, axis=0
    )

    dataset.test = np.delete(dataset.test, del_test_list, axis=0)
    dataset.labels_test = np.delete(dataset.labels_test, del_test_list, axis=0)
    dataset.test_idx = np.delete(dataset.test_idx, del_test_list, axis=0)

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


@dataclass
class RuleInfo:
    """Rule information"""

    rule_str: list[str]
    precision: float
    coverage: float


def plot_weights(
    weights: list[float],
    feature_names: list[str],
    rule_info: RuleInfo | None = None,
    img_name: str | None = None,
) -> None:
    """Plot the weights of the surrogate model.

    Parameters
    ----------
    weights : list[float]
        The weights of the features
    feature_names : list[str]
        The names of the features
    anchor_str : str, optional
        The rule, by default None
    precision : float, optional
        The precision, by default None
    coverage : float, optional
        The coverage, by default None
    img_name : str, optional
        The name of the image, by default None

    Returns
    -------
    None
    """

    features = feature_names
    abs_values = [abs(x) for x in weights]
    _, sorted_features, sorted_values = zip(
        *sorted(zip(abs_values, features, weights), reverse=False)[-5:]
    )
    plt.figure()
    color = [
        "#32a852" if sorted_values[i] > 0 else "#cf4529"
        for i in range(len(sorted_values))
    ]
    plt.barh(sorted_features, sorted_values, color=color)

    def concat_names(names: list[str]) -> str:
        """concatenate the names to multiline string"""
        multiline_names = []
        max_i = int(len(names) / 3)
        for i in range(max_i):
            triple = [names[i * 3], names[i * 3 + 1], names[i * 3 + 2]]
            multiline_names.append(" AND ".join(triple))
        if len(names) != max_i * 3:
            multiline_names.append(" AND ".join(names[max_i * 3 :]))
        return " AND \n".join(multiline_names)

    if rule_info is not None:
        anchor_str = concat_names(rule_info.rule_str)
        plt.title(
            f"{anchor_str}\n"
            f"with Precision {rule_info.precision:.3f} "
            f"and Coverage {rule_info.coverage:.3f}"
        )

    for f, v in zip(sorted_features, sorted_values):
        plt.text(v, f, round(v, 5))

    if img_name is not None:
        plt.savefig(img_name, bbox_inches="tight")

    plt.close()


def lime_original(
    trg: IntArray,
    pred_label: int,
    dataset: Dataset,
    model: sklearn.ensemble.RandomForestClassifier,
) -> list[float]:
    """Run LIME and plot the weights.

    Parameters
    ----------
    trg : np.ndarray
        The target sample
    pred_label : int
        The predicted label of the target sample
    dataset : Dataset
        The dataset
    model : sklearn.ensemble.RandomForestClassifier
        The black box model (random forest)

    Returns
    -------
    list[float]
        The weights of the features
    """

    lime_explainer = lime_tabular.LimeTabularExplainer(
        dataset.train,
        feature_names=dataset.feature_names,
        class_names=dataset.class_names,
        discretize_continuous=False,
    )
    lime_exp = lime_explainer.explain_instance(
        trg, model.predict_proba, num_features=5, top_labels=1
    )
    weights = [0.0] * len(dataset.feature_names)
    for t in lime_exp.local_exp[pred_label]:
        weights[t[0]] = t[1] * (pred_label * 2 - 1)
    return weights


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
