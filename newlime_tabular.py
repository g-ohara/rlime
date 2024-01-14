"""NewLIME for tabular datasets

This module implements NewLIME explainer for tabular datasets.
"""

from dataclasses import dataclass

import newlime_base
from newlime_base import Arm, Classifier, HyperParam, IntArray

Mapping = dict[int, tuple[int, str, int]]


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


def get_ordinal_ranges(
    arm: Arm, mapping: Mapping
) -> dict[int, list[int | None]]:
    """Get ordinal ranges for each feature in arm"""
    ordinal_ranges: dict[int, list[int | None]] = {}
    for idx in arm.rule:
        f, op, v = mapping[idx]
        if op in ("geq", "leq"):
            if f not in ordinal_ranges:
                ordinal_ranges[f] = [None, None]
        if op == "geq":
            trg_val = ordinal_ranges[f][0]
            if trg_val is None:
                ordinal_ranges[f][0] = v
            else:
                ordinal_ranges[f][0] = max(trg_val, v)
        if op == "leq":
            trg_val = ordinal_ranges[f][1]
            if trg_val is None:
                ordinal_ranges[f][1] = v
            else:
                ordinal_ranges[f][1] = min(trg_val, v)

    return ordinal_ranges


def add_names_to_exp(
    arm: Arm,
    mapping: Mapping,
    feature_names: list[str],
    categorical_names: dict[int, list[str]],
) -> list[str]:
    """Add names to the explanation

    Parameters
    ----------
    arm: Arm
        The arm to be explained.
    mapping: Mapping
        Mapping from index to feature name.

    Returns
    -------
    tuple[list[str], list[int]]
        The names and features of the arm.
    """

    names = []

    ordinal_ranges = get_ordinal_ranges(arm, mapping)
    handled = set()
    for idx in arm.rule:
        f, op, v = mapping[idx]
        # v = data_row[f]
        if op == "eq":
            fname = f"{feature_names[f]} = "
            if f in categorical_names:
                v = int(v)
                if (
                    "<" in categorical_names[f][v]
                    or ">" in categorical_names[f][v]
                ):
                    fname = ""
                fname = f"{fname}{categorical_names[f][v]}"
            else:
                fname = f"{fname}{v:.2f}"
        else:
            if f in handled:
                continue

            def get_vals(f: int) -> str:
                geq, leq = ordinal_ranges[f]
                geq_val = ""
                leq_val = ""
                if geq is not None:
                    name = categorical_names[f][geq + 1]
                    if "<" in name:
                        geq_val = name.split()[0]
                    elif ">" in name:
                        geq_val = name.split()[-1]
                if leq is not None:
                    name = categorical_names[f][leq]
                    if leq == 0:
                        leq_val = name.split()[-1]
                    elif "<" in name:
                        leq_val = name.split()[-1]

                fname = ""
                if leq_val and geq_val:
                    fname = f"{geq_val} < {feature_names[f]} <= {leq_val}"
                elif leq_val:
                    fname = f"{feature_names[f]} <= {leq_val}"
                elif geq_val:
                    fname = f"{feature_names[f]} > {geq_val}"
                return fname

            fname = get_vals(f)
            handled.add(f)
        names.append(fname)
    return names


def get_mapping(
    trg: IntArray,
    categorical_features: list[int],
    ordinal_features: list[int],
    categorical_names: dict[int, list[str]],
) -> Mapping:
    """Get mapping from index to feature name

    Parameters
    ----------
    trg: IntArray
        Target instance.
    categorical_features: list[int]
    ordinal_features: list[int]
    categorical_names: dict[int, list[str]]

    Returns
    -------
    Mapping
        Dicrionary from index to feature, operator, and value.
        - feature: int
        - operator: str (one of "eq", "geq" or "leq")
        - value: int
    """
    mapping: Mapping = {}
    for f in categorical_features:
        if f in ordinal_features:
            for v in range(len(categorical_names[f])):
                idx = len(mapping)
                if trg[f] <= v:
                    mapping[idx] = (f, "leq", v)
                elif trg[f] > v:
                    mapping[idx] = (f, "geq", v)
        else:
            idx = len(mapping)
            mapping[idx] = (f, "eq", trg[f])
    return mapping


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
    sampler = newlime_base.Sampler(
        data_row, dataset.data, classifier_fn, dataset.categorical_names
    )
    arm = newlime_base.beam_search(sampler, hyper_param)
    if arm is None:
        return None

    # Generate Mapping
    mapping = get_mapping(
        data_row,
        dataset.categorical_features,
        dataset.ordinal_features,
        dataset.categorical_names,
    )

    names = add_names_to_exp(
        arm, mapping, dataset.feature_names, dataset.categorical_names
    )
    return names, arm
