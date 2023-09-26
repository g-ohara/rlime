import my_anchor_base

from anchor import anchor_tabular
from anchor import anchor_explanation

import numpy as np
import pandas as pd

from river import compose

from typing import Callable
from typing import Type
from typing import cast 
from typing import Any 

Predicate = tuple[int, str, int]
Rule = dict[int, Predicate]
Classifier = Callable[[np.ndarray], np.ndarray]
Surrogate = Type[compose.Pipeline]
SampleFn = Callable[
        [Rule, int, bool, Surrogate | None, bool], 
        tuple[np.ndarray, np.ndarray, np.ndarray]]

class AnchorTabularExplainer(anchor_tabular.AnchorTabularExplainer):
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
            self, 
            data_row        : np.ndarray,
            classifier_fn   : Classifier,
            desired_label   : int | None = None
            ) -> tuple[
                    SampleFn, 
                    Rule]:
        
        def predict_fn(x: np.ndarray) -> np.ndarray:
            return classifier_fn(self.encoder_fn(x))
        ##

        true_label: int | None = desired_label
        if true_label is None:
            # set true_label same as the label of data_row
            true_label = predict_fn(data_row.reshape(1, -1))[0]

        # must map present here to include categorical features 
        # (for conditions_eq), and numerical features for geq and leq
        mapping : Rule = {} 

        # 説明したいインスタンスを離散化
        data_row = self.disc.discretize(data_row.reshape(1, -1))[0]

        # 任意のカテゴリカル特徴量 f について
        for f in self.categorical_features:
            # f が順序特徴量ならば
            if f in self.ordinal_features:
                for v in range(len(self.categorical_names[f])):
                    idx = len(mapping)
                    if data_row[f] <= v and v != len(self.categorical_names[f]) - 1:
                        mapping[idx] = (f, 'leq', v)
                        # names[idx] = '%s <= %s' % (self.feature_names[f], v)
                    elif data_row[f] > v:
                        mapping[idx] = (f, 'geq', v)
                        # names[idx] = '%s > %s' % (self.feature_names[f], v)
            # f が非順序特徴量ならば
            else:
                idx = len(mapping)
                mapping[idx] = (f, 'eq', data_row[f])
            # names[idx] = '%s = %s' % (
            #     self.feature_names[f],
            #     self.categorical_names[f][int(data_row[f])])
        ## end for

        # *********************************************************************
        # Modified Points
        # 
        # 1. Added some arguments (surrogate_model, update_model)
        # 2. Changed the way to compute labels 
        # 
        def sample_fn(
                present         : Rule, 
                num_samples     : int, 
                compute_labels  : bool = True,
                surrogate_model : Surrogate | None = None,
                update_model    : bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

            conditions_eq : dict[int, int] = {}
            conditions_leq: dict[int, int] = {}
            conditions_geq: dict[int, int] = {}
            for x in present:
                f, op, v = mapping[x]
                if op == 'eq':
                    conditions_eq[f] = v
                if op == 'leq':
                    if f not in conditions_leq:
                        conditions_leq[f] = v
                    conditions_leq[f] = min(conditions_leq[f], v)
                if op == 'geq':
                    if f not in conditions_geq:
                        conditions_geq[f] = v
                    conditions_geq[f] = max(conditions_geq[f], v)
            # conditions_eq = dict([(x, data_row[x]) for x in present])

            # Sample new data points
            raw_data = self.sample_from_train(
                conditions_eq, {}, conditions_geq, conditions_leq, num_samples)

            # Descretize new data points
            d_raw_data = self.disc.discretize(raw_data)

            # Restore original representations for new data points
            data = np.zeros((num_samples, len(mapping)), int)
            for i in mapping:
                f, op, v = mapping[i]
                if op == 'eq':
                    data[:, i] = (d_raw_data[:, f] == data_row[f]).astype(int)
                if op == 'leq':
                    data[:, i] = (d_raw_data[:, f] <= v).astype(int)
                if op == 'geq':
                    data[:, i] = (d_raw_data[:, f] > v).astype(int)
            # data = (raw_data == data_row).astype(int)

            # *****************************************************************
            labels = np.array([])
            if surrogate_model != None:
                given_model: Surrogate = cast(Surrogate, surrogate_model)
                data_x: pd.DataFrame = pd.DataFrame(raw_data)
                data_y = pd.Series(predict_fn(raw_data))
                if compute_labels:
                    labels = (given_model.predict_many(data_x) == data_y).astype(int)
                if update_model:
                    given_model.learn_many(data_x, data_y)
            # *****************************************************************

            return raw_data, data, labels
        ## end function

        return sample_fn, mapping

    def explain_instance(
            self, 
            data_row        : np.ndarray, 
            classifier_fn   : Classifier, 
            threshold       : float = 0.95,
            delta           : float = 0.1, 
            tau             : float = 0.15, 
            batch_size      : int = 100,
            max_anchor_size : int | None = None,
            desired_label   : int | None = None,
            beam_size       : int = 4, 
            **kwargs        : Any) -> tuple[anchor_explanation.AnchorExplanation, Surrogate]:
        
        # サンプリングのための関数を取得
        # sample_fn --- 摂動サンプルとその擬似ラベルを返却する関数
        # It's possible to pass in max_anchor_size
        sample_fn, mapping = self.get_sample_fn(
            data_row, classifier_fn, desired_label=desired_label)
        # return sample_fn, mapping

        # *********************************************************************
        # Generate Explanation
        exp, surrogate_model = my_anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, 
            delta=delta, 
            epsilon=tau, 
            batch_size=batch_size,
            desired_confidence=threshold, 
            max_anchor_size=max_anchor_size,
            verbose=False,
            **kwargs)
        # *********************************************************************
        
        self.add_names_to_exp(data_row, exp, mapping)
        exp['instance'] = data_row
        exp['prediction'] = classifier_fn(
                self.encoder_fn(data_row.reshape(1, -1)))[0]
        explanation = anchor_explanation.AnchorExplanation(
                'tabular', exp, self.as_html)

        # *********************************************************************
        return explanation, surrogate_model
        # *********************************************************************
