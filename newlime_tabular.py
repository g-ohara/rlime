"""NewLIME for tabular datasets

This module implements NewLIME explainer for tabular datasets.
"""

import dataclasses
from typing import Callable

import numpy as np
import pandas as pd
from anchor import anchor_tabular
from river import compose

import newlime_base
from newlime_base import Arm, HyperParam, NewLimeBaseBeam, SampleFn, Samples

Classifier = Callable[[np.ndarray], np.ndarray]
Mapping = dict[int, tuple[int, str, int]]


@dataclasses.dataclass
class Conditions:
    """Class for conditions of equality and inequality.

    Attributes
    ----------
    eq : dict[int, int]
        Equality conditions.
    leq : dict[int, int]
        Inequality conditions (<=).
    geq : dict[int, int]
        Inequality conditions (>=).
    """

    eq: dict[int, int]
    leq: dict[int, int]
    geq: dict[int, int]


class NewLimeTabularExplainer(anchor_tabular.AnchorTabularExplainer):
    """NewLIME explainer for tabular datasets.

    Attributes
    ----------
    categorical_features: list of int
        indices of categorical features
    ordinal_features: list of int
        indices of ordinal categorical features

    Parameters
    ----------
    training_data: np.ndarray
        training data
    encoder_fn: callable
        function that takes raw data and returns encoded data
    disc: anchor.discretize.BaseDiscretizer
        discretizer for continuous features

    Keyword Arguments
    -----------------
    categorical_names: dict[int, list[str]]
        map from integer to list of strings, names for each value of the
        categorical features. Every feature that is not in this map will be
        considered as ordinal or continuous, and thus discretized.
    """

    @staticmethod
    def get_conditions(
        present: newlime_base.Rule, mapping: Mapping
    ) -> Conditions:
        """Classify conditions in given rule into eqality, inequality(<) and
        inequality(>).

        Parameters
        ----------
        present : tuple[int, ...]
            Target rule.
        mapping : dict[int, tuple[int, str, int]]
            Set of conditions in target instance (data_row).
            f : int
                index of feature
            op : str
                comparison operator
            v : int
                index of categorical name

        Returns
        -------
        Conditions
            Classify conditions in given rule into eqality, inequality(<) and
            inequality(>).
        """

        conditions_eq: dict[int, int] = {}
        conditions_leq: dict[int, int] = {}
        conditions_geq: dict[int, int] = {}

        x: int  # a predicate index in given rule
        for x in present:
            f: int  # feature index (f == x ??)
            op: str  # comparison operator
            v: int  # standard value
            f, op, v = mapping[x]
            if op == "eq":
                conditions_eq[f] = v
            elif op == "leq":
                if f not in conditions_leq:
                    conditions_leq[f] = v
                conditions_leq[f] = min(conditions_leq[f], v)
            elif op == "geq":
                if f not in conditions_geq:
                    conditions_geq[f] = v
                conditions_geq[f] = max(conditions_geq[f], v)
        # conditions_eq = dict([(x, data_row[x]) for x in present])
        return Conditions(conditions_eq, conditions_leq, conditions_geq)

    @staticmethod
    def restore_data(
        data_row: np.ndarray,
        d_raw_data: np.ndarray,
        num_samples: int,
        mapping: Mapping,
    ) -> np.ndarray:
        """Restore original representations for new data points.

        Parameters
        ----------
        data_row : np.ndarray
            Target instance.
        d_raw_data : np.ndarray
            Discretized data points (Discretized raw_data).
        num_samples : int
            The number of returned sample.
        mapping : dict[int, tuple[int, str, int]]
            Set of conditions in target instance (data_row).
            f : int
                index of feature
            op : str
                comparison operator
            v : int
                index of categorical name

        Returns
        -------
        np.ndarray
            Original representations for new data points.
        """

        data: np.ndarray = np.zeros((num_samples, len(mapping)), int)
        for i, predicate in mapping.items():
            f, op, v = predicate
            if op == "eq":
                data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)
            if op == "leq":
                data[:, i] = (d_raw_data[:, f] <= v).astype(int)
            if op == "geq":
                data[:, i] = (d_raw_data[:, f] > v).astype(int)
        # data = (raw_data == data_row).astype(int)
        return data

    def get_sample_fn(
        self,
        data_row: np.ndarray,  # target instance
        classifier_fn: Classifier,
        desired_label: int | None = None,
    ) -> tuple[SampleFn, Mapping]:
        """Return sampling function that gets sample from neighborhood of
        target instance. The function returns ......

        Parameters
        ----------
        data_row : np.ndarray
            Target instance.
        classifier_fn: Classifier
            Blackbox classifier that labels new data points.
        desired_label: int | None, default None
            Label that new data points are required to have.

        Returns
        -------
        sample_fn : SampleFn
        mapping : dict[int, tuple[int, str, int]]
            Set of conditions in target instance (data_row).
            f : int
                index of feature
            op : str
                comparison operator
            v : int
                index of categorical name
        """

        def predict_fn(x: np.ndarray) -> np.ndarray:
            """Get predictions of the blackbox classifier"""
            if x.shape[0] == 0:
                return np.array([])
            return classifier_fn(self.encoder_fn(x))

        # must map present here to include categorical features
        # (for conditions_eq), and numerical features for geq and leq
        mapping: Mapping = {}

        # Discritize the target instance.
        data_row = self.disc.discretize(data_row.reshape(1, -1))[0]

        for f in self.categorical_features:
            if f in self.ordinal_features:
                # Add f to mapping as inequality condition if f is ordinal
                # categorical feature.
                for v in range(len(self.categorical_names[f])):
                    idx = len(mapping)
                    if (
                        data_row[f] <= v
                        and v != len(self.categorical_names[f]) - 1
                    ):
                        mapping[idx] = (f, "leq", v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, "geq", v)
            else:
                # Add f to mapping as equality condition if f is non-ordinal
                # categorical feature.
                idx = len(mapping)
                mapping[idx] = (f, "eq", int(data_row[f]))
        ##

        # *********************************************************************
        # Modified Points
        #
        # 1. Added some arguments (surrogate_model, update_model)
        # 2. Changed the way to compute labels
        #
        def sample_fn(
            num_samples: int,
            present: newlime_base.Rule,
            compute_labels: bool = True,
            surrogate_model: compose.Pipeline | None = None,
            update_model: bool = True,
        ) -> Samples:
            """Get sample satisfying given rule and its label predicted by
            the surrogate model, then update the surrogate model for the sample

            Parameters
            ----------
            present : tuple[int, ...]
                Target rule.
            num_samples : int
                The number of returned sample.
            compute_labels : bool default True
                Whether compute reward or not
            surrogate_model: compose.Pipeline | None default None
                The surrogate model that will be updated on sampling.
                If None, it does not compute reward or update the model
                regardless of the value of surrogate_model and update_model.
            update_model : bool default True
                Whether update the surrogate model on sampling or not

            Returns
            -------
            Sample
                Data points sampled from neighborhood of the target rule
                (present).
                raw_data : np.ndarray
                    Original data points sampled under the target rule
                    (present).
                d_raw_data : np.ndarray
                    Discretized data points (Discretized raw_data).
                data : np.ndarray
            """

            # Return empty data if the number of samples is 0
            if num_samples == 0:
                return Samples(np.array([]), np.array([]), np.array([]))

            conditions = NewLimeTabularExplainer.get_conditions(
                present, mapping
            )

            # Sample new data points (sample_from_train)
            # 1. Get samples randomly from train set
            # 2. Change the values of each sample to match the given conditions
            # NOTE: This proceidure assumes that the features are independent
            # of each other, but this assumption does not hold in general.
            raw_data: np.ndarray = self.sample_from_train(
                conditions.eq, {}, conditions.geq, conditions.leq, num_samples
            )

            # Descretize new data points
            d_raw_data: np.ndarray = self.disc.discretize(raw_data)

            # Restore original representations for new data points
            data = NewLimeTabularExplainer.restore_data(
                data_row, d_raw_data, num_samples, mapping
            )

            # *****************************************************************
            rewards: np.ndarray = np.array([])
            if surrogate_model is not None:
                data_x: pd.DataFrame = pd.DataFrame(raw_data)
                data_y = pd.Series(predict_fn(raw_data))
                if compute_labels:
                    rewards = (
                        surrogate_model.predict_many(data_x) == data_y
                    ).astype(int)
                if update_model:
                    surrogate_model.learn_many(data_x, data_y)
            # *****************************************************************

            return Samples(raw_data, data, rewards)

        ##

        return sample_fn, mapping

    def my_add_names_to_exp(
        self, arm: Arm, mapping: Mapping
    ) -> tuple[list[str], list[int]]:
        names = []
        feature = [mapping[idx][0] for idx in arm.rule]

        def get_ordinal_ranges(arm: Arm) -> dict[int, list[float]]:
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

        ordinal_ranges = get_ordinal_ranges(arm)
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
                geq, leq = ordinal_ranges[f]
                fname = ""
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
                if leq_val and geq_val:
                    fname = f"{geq_val} < {self.feature_names[f]} <= {leq_val}"
                elif leq_val:
                    fname = f"{self.feature_names[f]} <= {leq_val}"
                elif geq_val:
                    fname = f"{self.feature_names[f]} > {geq_val}"
                handled.add(f)
            names.append(fname)
        return names, feature

    def my_explain_instance(
        self,
        data_row: np.ndarray,
        classifier_fn: Classifier,
        hyper_param: HyperParam,
    ) -> tuple[list[str], list[int], Arm] | None:
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
        sample_fn, mapping = self.get_sample_fn(data_row, classifier_fn)

        # Generate Explanation
        arm: Arm | None
        arm = NewLimeBaseBeam.beam_search(sample_fn, hyper_param)
        if arm is None:
            return None

        names, feature = self.my_add_names_to_exp(arm, mapping)
        return names, feature, arm
