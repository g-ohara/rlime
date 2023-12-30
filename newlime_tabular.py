"""NewLIME for tabular datasets

This module implements NewLIME explainer for tabular datasets.
"""

import dataclasses
from typing import Callable

import anchor.utils
import numpy as np
from anchor import anchor_tabular

import newlime_base
from newlime_base import Arm, HyperParam, NewLimeBaseBeam

Classifier = Callable[[np.ndarray], np.ndarray]
Mapping = dict[int, tuple[int, str, int]]


@dataclasses.dataclass
class Dataset(
    anchor.utils.Bunch
):  # pylint: disable=too-many-instance-attributes
    """Dataset class"""

    data: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    train: np.ndarray
    labels_train: np.ndarray
    validation_idx: np.ndarray
    validation: np.ndarray
    labels_validation: np.ndarray
    test_idx: np.ndarray
    test: np.ndarray
    labels_test: np.ndarray
    feature_names: list[str]
    categorical_names: dict[int, list[str]]
    class_target: str
    class_names: list[str]


class NewLimeTabularExplainer(anchor_tabular.AnchorTabularExplainer):
    """NewLIME explainer for tabular datasets."""

    @staticmethod
    def get_ordinal_ranges(
        arm: Arm, mapping: Mapping
    ) -> dict[int, list[float]]:
        """Get ordinal ranges for each feature in arm"""
        ordinal_ranges = {}
        for idx in arm.rule:
            f, op, v = mapping[idx]
            if op in ("geq", "leq"):
                if f not in ordinal_ranges:
                    ordinal_ranges[f] = [float("-inf"), float("inf")]
            if op == "geq":
                ordinal_ranges[f][0] = max(ordinal_ranges[f][0], v)
            if op == "leq":
                ordinal_ranges[f][1] = min(ordinal_ranges[f][1], v)
        return ordinal_ranges

    def my_add_names_to_exp(self, arm: Arm, mapping: Mapping) -> list[str]:
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

        ordinal_ranges = self.get_ordinal_ranges(arm, mapping)
        handled = set()
        for idx in arm.rule:
            f, op, v = mapping[idx]
            # v = data_row[f]
            if op == "eq":
                fname = f"{self.feature_names[f]} = "
                if f in self.categorical_names:
                    v = int(v)
                    if (
                        "<" in self.categorical_names[f][v]
                        or ">" in self.categorical_names[f][v]
                    ):
                        fname = ""
                    fname = f"{fname}{self.categorical_names[f][v]}"
                else:
                    fname = f"{fname}{v:.2f}"
            else:
                if f in handled:
                    continue

                def get_vals(f: int) -> str:
                    geq, leq = ordinal_ranges[f]
                    geq_val = ""
                    leq_val = ""
                    if geq > float("-inf"):
                        name = self.categorical_names[f][geq + 1]
                        if "<" in name:
                            geq_val = name.split()[0]
                        elif ">" in name:
                            geq_val = name.split()[-1]
                    if leq < float("inf"):
                        name = self.categorical_names[f][leq]
                        if leq == 0:
                            leq_val = name.split()[-1]
                        elif "<" in name:
                            leq_val = name.split()[-1]

                    fname = ""
                    if leq_val and geq_val:
                        fname = (
                            f"{geq_val} < {self.feature_names[f]} <= {leq_val}"
                        )
                    elif leq_val:
                        fname = f"{self.feature_names[f]} <= {leq_val}"
                    elif geq_val:
                        fname = f"{self.feature_names[f]} > {geq_val}"
                    return fname

                fname = get_vals(f)
                handled.add(f)
            names.append(fname)
        return names

    def my_explain_instance(
        self,
        data_row: np.ndarray,
        dataset: Dataset,
        classifier_fn: Classifier,
        hyper_param: HyperParam,
    ) -> tuple[list[str], Arm] | None:
        """Generate NewLIME explanation for given classifier on neighborhood of
        given data point.

        Parameters
        ----------
        data_row : np.ndarray
            Target instance.
        classifier_fn: Classifier
            Blackbox classifier that labels new data points.
        hyper_param: HyperParam
            Hyperparameters for NewLIME.

        Returns
        -------
        tuple[anchor_explanation.AnchorExplanation, compose.Pipeline | None] |
        None
            The explanation and the surrogate model.
        """

        # It's possible to pass in max_anchor_size
        _, mapping = self.get_sample_fn(data_row, classifier_fn)

        # Generate Explanation
        arm: Arm | None
        sampler = newlime_base.Sampler(
            data_row, dataset.data, classifier_fn, dataset.categorical_names
        )
        arm = NewLimeBaseBeam.beam_search(sampler, hyper_param)
        if arm is None:
            return None

        names = self.my_add_names_to_exp(arm, mapping)
        return names, arm
