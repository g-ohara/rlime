from typing import Any, cast

import numpy as np
import pandas as pd
from anchor import anchor_explanation, anchor_tabular
from river import compose

from newlime_base import (Anchor, Classifier, Mapping, NewLimeBaseBeam, Sample,
                          SampleFn)


class NewLimeTabularExplainer(anchor_tabular.AnchorTabularExplainer):
    """
    Args:
        class_names: list of strings
        feature_names: list of strings
        train_data: used to sample (bootstrap)
        categorical_names: map from integer to list of strings, names for each
            value of the categorical features. Every feature that is not in
            this map will be considered as ordinal or continuous, and thus
            discretized.
    """

    def get_sample_fn(
        self, data_row: np.ndarray, classifier_fn: Classifier
    ) -> tuple[SampleFn, Mapping]:
        # Get predictions of the blackbox classifier
        def predict_fn(x: np.ndarray) -> np.ndarray:
            return classifier_fn(self.encoder_fn(x))

        ##

        # must map present here to include categorical features
        # (for conditions_eq), and numerical features for geq and leq
        mapping: Mapping = {}

        data_row = self.disc.discretize(data_row.reshape(1, -1))[0]

        for f in self.categorical_features:
            if f in self.ordinal_features:
                for v in range(len(self.categorical_names[f])):
                    idx = len(mapping)
                    if (
                        data_row[f] <= v
                        and v != len(self.categorical_names[f]) - 1
                    ):
                        mapping[idx] = (f, "leq", v)
                        # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, "geq", v)
                        # names[idx] = '%s > %s' % (self.feature_names[f], v)
            else:
                idx = len(mapping)
                mapping[idx] = (f, "eq", int(data_row[f]))
            # names[idx] = '%s = %s' % (
            #     self.feature_names[f],
            #     self.categorical_names[f][int(data_row[f])])
        ##

        # *********************************************************************
        # Modified Points
        #
        # 1. Added some arguments (surrogate_model, update_model)
        # 2. Changed the way to compute labels
        #
        def sample_fn(
            present: list[int],
            num_samples: int,
            compute_labels: bool = True,
            surrogate_model: compose.Pipeline | None = None,
            update_model: bool = True,
        ) -> Sample:
            conditions_eq: dict[int, int] = {}
            conditions_leq: dict[int, int] = {}
            conditions_geq: dict[int, int] = {}
            for x in present:
                f, op, v = mapping[x]
                if op == "eq":
                    conditions_eq[f] = v
                if op == "leq":
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == "geq":
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            # conditions_eq = dict([(x, data_row[x]) for x in present])

            # Sample new data points
            raw_data: np.ndarray = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples
            )

            # Descretize new data points
            d_raw_data: np.ndarray = self.disc.discretize(raw_data)

            # Restore original representations for new data points
            data: np.ndarray = np.zeros((num_samples, len(mapping)), int)

            for i in mapping:
                f, op, v = mapping[i]
                if op == "eq":
                    data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)
                if op == "leq":
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == "geq":
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)
            # data = (raw_data == data_row).astype(int)

            # *****************************************************************
            labels: np.ndarray = np.array([])
            if surrogate_model is not None:
                given_model: compose.Pipeline = cast(
                    compose.Pipeline, surrogate_model
                )
                data_x: pd.DataFrame = pd.DataFrame(raw_data)
                data_y = pd.Series(predict_fn(raw_data))
                if compute_labels:
                    labels = (
                        given_model.predict_many(data_x) == data_y
                    ).astype(int)
                if update_model:
                    given_model.learn_many(data_x, data_y)
            # *****************************************************************

            return Sample(raw_data, data, labels)

        ##

        return sample_fn, mapping

    def explain_instance(
        self,
        data_row: np.ndarray,
        classifier_fn: Classifier,
        threshold: float = 0.95,
        delta: float = 0.1,
        epsilon: float = 0.15,
        batch_size: int = 100,
        beam_size: int = 4,
        verbose: bool = False,
        my_verbose: bool = False,
        **kwargs: Any
    ) -> (
        tuple[anchor_explanation.AnchorExplanation, compose.Pipeline | None]
        | None
    ):
        # サンプリングのための関数を取得
        # sample_fn --- 摂動サンプルとその擬似ラベルを返却する関数
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(data_row, classifier_fn)
        # return sample_fn, mapping

        # *********************************************************************
        # Generate Explanation
        result: tuple[Anchor, compose.Pipeline | None] | None
        result = NewLimeBaseBeam.beam_search(
            sample_fn,
            delta=delta,
            epsilon=epsilon,
            batch_size=batch_size,
            desired_confidence=threshold,
            beam_size=beam_size,
            verbose=verbose,
            my_verbose=my_verbose,
            **kwargs
        )
        # *********************************************************************

        if result is None:
            return None

        exp, surrogate_model = result
        self.add_names_to_exp(data_row, exp, mapping)
        exp["instance"] = data_row
        exp["prediction"] = classifier_fn(
            self.encoder_fn(data_row.reshape(1, -1))
        )[0]
        explanation = anchor_explanation.AnchorExplanation(
            "tabular", exp, self.as_html
        )
        return explanation, surrogate_model

    ##
