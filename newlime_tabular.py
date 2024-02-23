"""NewLIME for tabular datasets

This module implements NewLIME explainer for tabular datasets.
"""

from dataclasses import dataclass

import newlime_base
from newlime_base import Arm, HyperParam
from newlime_types import Classifier, IntArray, Rule
from sampler import Sampler


@dataclass
class Dataset:  # pylint: disable=too-many-instance-attributes
    """Dataset class"""

    data: IntArray
    labels: IntArray
    train_idx: IntArray
    train: IntArray
    labels_train: IntArray
    validation_idx: IntArray
    validation: IntArray
    labels_validation: IntArray
    test_idx: IntArray
    test: IntArray
    labels_test: IntArray
    feature_names: list[str]
    categorical_features: list[int]
    categorical_names: dict[int, list[str]]
    ordinal_features: list[int]
    class_target: str
    class_names: list[str]


def add_names_to_exp(
    trg: IntArray,
    rule: Rule,
    feature_names: list[str],
    ordinal_features: list[int],
    categorical_names: dict[int, list[str]],
) -> list[str]:
    """Converts the rule to a string representation.

    Parameters
    ----------
    trg: IntArray
        Target instance.
    rule: Rule
        The rule.
    feature_names: list[str]
        The names of the features.
    ordinal_features: list[int]
        The ordinal features.
    categorical_names: dict[int, list[str]]
        The names of the categorical features.

    Returns
    -------
    list[str]
        The string representation of the rule.
    """

    names = []
    for i in rule:
        if i in ordinal_features:
            name = categorical_names[i][int(trg[i])]
        else:
            name = feature_names[i] + " = " + categorical_names[i][int(trg[i])]
        names.append(name)
    return names


def explain_instance(
    data_row: IntArray,
    dataset: Dataset,
    classifier_fn: Classifier,
    hyper_param: HyperParam,
) -> tuple[list[str], Arm] | None:
    """Generate NewLIME explanation for given classifier on neighborhood
    of given data point.

    Parameters
    ----------
    data_row : IntArray
        Target instance.
    classifier_fn: Classifier
        Blackbox classifier that labels new data points.
    hyper_param: HyperParam
        Hyperparameters for NewLIME.

    Returns
    -------
    tuple[list[str], Arm] | None
        The string representation of the rule and the arm.
    """

    # Generate Explanation
    arm: Arm | None
    sampler = Sampler(
        data_row, dataset.data, classifier_fn, dataset.categorical_names
    )
    arm = newlime_base.beam_search(sampler, hyper_param)
    if arm is None:
        return None

    # Get names from the rule
    names = add_names_to_exp(
        data_row,
        arm.rule,
        dataset.feature_names,
        dataset.ordinal_features,
        dataset.categorical_names,
    )

    return names, arm
