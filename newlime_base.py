"""Base functions"""

from __future__ import print_function

import collections
import copy
import dataclasses
import functools
from typing import Any, Callable

import numpy as np
from anchor.anchor_base import AnchorBaseBeam
from river import compose, linear_model, preprocessing

Rule = tuple[int, ...]
Classifier = Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass
class Sample:
    """This is a class for perturbed vectors sampled from distribution"""

    raw_data: np.ndarray
    data: np.ndarray
    labels: np.ndarray


SampleFn = Callable[
    [list[int], int, bool, compose.Pipeline | None, bool], Sample
]
CompleteSampleFn = Callable[[int], float]
State = dict[str, Any]
Anchor = dict[str, Any]


@dataclasses.dataclass
class RuleClass:
    """This is the class for a rule, model learned under that rule,
    precision of that model and coverage of that rule"""

    rule: Rule
    model: compose.Pipeline | None
    precision: float | None
    coverage: float


@dataclasses.dataclass
class HyperParam:
    """Hyper parameters for beam search"""

    delta: float = 0.05
    epsilon: float = 0.1
    epsilon_stop: float = 0.05
    beam_size: int = 10
    batch_size: int = 10
    desired_confidence: float = 1.0
    coverage_samples_num: int = 10000


# class Arm:
#     def __init__(self, rule: Rule, sample_fn: SampleFn) -> None:
#         self.rule = rule
#         self.model = compose.Pipeline(
#             preprocessing.StandardScaler(), linear_model.LogisticRegression()
#         )
#         self.sample_fn = sample_fn
#
#     def get_reward(self, sample_num: int, state: State) -> float:
#         return NewLimeBaseBeam.complete_sample_fn(
#             self.sample_fn, self.rule, sample_num, self.model, state
#         )


class NewLimeBaseBeam:
    """This is a class for beam search of best anchor"""

    @staticmethod
    def update_state(
        state: State, rule: Rule, batch_size: int, sample: Sample
    ) -> None:
        """Update state after sampling"""
        current_idx: int
        current_idx = state["current_idx"]

        idxs: range
        idxs = range(current_idx, current_idx + batch_size)

        if "<U" in str(sample.raw_data.dtype):
            # String types: make sure both string types are of maximum length
            # to avoid string truncation. E.g., '<U308', '<U290' -> '<U308'
            max_dtype = max(
                str(state["raw_data"].dtype), str(sample.raw_data.dtype)
            )
            state["raw_data"] = state["raw_data"].astype(max_dtype)
            sample.raw_data = sample.raw_data.astype(max_dtype)

        state["t_idx"][rule].update(idxs)
        state["t_nsamples"][rule] += batch_size
        state["t_positives"][rule] += sample.labels.sum()
        state["data"][idxs] = sample.data
        state["raw_data"][idxs] = sample.raw_data
        if len(sample.labels) > 0:
            state["labels"][idxs] = sample.labels
        state["current_idx"] += batch_size
        if state["current_idx"] >= state["data"].shape[0] - max(
            1000, batch_size
        ):
            prealloc_size = state["prealloc_size"]
            current_idx = sample.data.shape[0]
            state["data"] = np.vstack(
                (
                    state["data"],
                    np.zeros(
                        (prealloc_size, sample.data.shape[1]),
                        sample.data.dtype,
                    ),
                )
            )
            state["raw_data"] = np.vstack(
                (
                    state["raw_data"],
                    np.zeros(
                        (prealloc_size, sample.raw_data.shape[1]),
                        sample.raw_data.dtype,
                    ),
                )
            )
            state["labels"] = np.hstack(
                (state["labels"], np.zeros(prealloc_size, sample.labels.dtype))
            )
        # This can be really slow
        # state['data'] = np.vstack((state['data'], data))
        # state['raw_data'] = np.vstack((state['raw_data'], raw_data))
        # state['labels'] = np.hstack((state['labels'], labels))

    ##

    @staticmethod
    def complete_sample_fn(
        sample_fn: SampleFn,
        rule: Rule,
        n: int,
        model: compose.Pipeline | None,
        state: State,
    ) -> float:
        """Sample perturbed vectors, update state and returns the reward"""

        # *****************************************************************
        # Take a sample satisfying the tuple t.
        sample: Sample
        sample = sample_fn(list(rule), n, True, model, True)
        # *****************************************************************

        NewLimeBaseBeam.update_state(state, rule, n, sample)

        ret: float = sample.labels.sum()
        return ret

    ##

    @staticmethod
    def init_surrogate_models(num_models: int) -> list[compose.Pipeline]:
        """Initialize online linear models and returns them"""
        models: list[compose.Pipeline] = []
        for _ in range(num_models):
            preprocessor = preprocessing.StandardScaler()
            model = linear_model.LogisticRegression()
            pipeline = compose.Pipeline(preprocessor, model)
            models.append(pipeline)
        return models

    ##

    @staticmethod
    def get_sample_fns(
        sample_fn: SampleFn,
        tuples: list[Rule],
        surrogate_models: list[compose.Pipeline],
        state: State,
    ) -> list[CompleteSampleFn]:
        """Return function that samples, updates model and returns sample"""
        # each sample fn returns number of instances that the surrogate model
        # classified correctly
        sample_fns: list[CompleteSampleFn] = []

        for rule, model in zip(tuples, surrogate_models):
            partial_fn = functools.partial(
                NewLimeBaseBeam.complete_sample_fn,
                sample_fn=sample_fn,
            )
            #     rule=rule,
            #     model=model,
            #     state=state,
            fn = lambda n, rule=rule, model=model, state=state: partial_fn(
                n=n, rule=rule, model=model, state=state
            )
            sample_fns.append(fn)

        return sample_fns

    ##

    @staticmethod
    def get_initial_statistics(tuples: list[Rule]) -> dict[str, np.ndarray]:
        """Returns statistics to pass to AnchorBaseBeam.lucb"""

        stats = {
            "n_samples": np.zeros(len(tuples)),
            "positives": np.zeros(len(tuples)),
        }
        return stats

    @staticmethod
    def b_best_cands(
        cands: list[Rule],
        sample_fn: SampleFn,
        hyper_param: HyperParam,
        state: State,
    ) -> tuple[
        list[int], list[Rule], list[CompleteSampleFn], list[compose.Pipeline]
    ]:
        """Search for B candidates with highest precision"""

        # list of the surrogate models under each rule
        # thus it has the same length as 'tuples'
        surrogate_models: list[compose.Pipeline]

        # Initialize surrogate models
        surrogate_models = NewLimeBaseBeam.init_surrogate_models(len(cands))

        # list of the functions to sample perturbed vectors under each rule
        # thus it has the same length as 'tuples'
        sample_fns: list[CompleteSampleFn]

        # Get sampling functions
        sample_fns = NewLimeBaseBeam.get_sample_fns(
            sample_fn, cands, surrogate_models, state
        )

        initial_stats = NewLimeBaseBeam.get_initial_statistics(cands)

        # list of the indexes to B candidate rules with the highest precision
        b_best_idxes: list[int]
        b_best_idxes = list(
            AnchorBaseBeam.lucb(
                sample_fns,
                initial_stats,
                hyper_param.epsilon,
                hyper_param.delta,
                hyper_param.batch_size,
                hyper_param.beam_size,
            )
        )

        # list of the B candidate rules with the highest precision
        # thus it has the same length as b_best_idxes
        b_best_cands: list[Rule]
        b_best_cands = [cands[i] for i in b_best_idxes]

        # Get candidates from their indexes
        return b_best_idxes, b_best_cands, sample_fns, surrogate_models

    ##

    @staticmethod
    def update_confidence_bound(
        rule: Rule, beta: float, state: State
    ) -> tuple[float, float, float]:
        """Update confidence bound of the precision of the rule based on
        state"""

        mean = state["t_positives"][rule] / state["t_nsamples"][rule]
        lb = AnchorBaseBeam.dlow_bernoulli(
            mean, beta / state["t_nsamples"][rule]
        )
        ub = AnchorBaseBeam.dup_bernoulli(
            mean, beta / state["t_nsamples"][rule]
        )

        return mean, lb, ub

    ##

    @staticmethod
    def largest_valid_cand(
        b_best_cands: list[tuple[int, Rule]],
        hyper_param: HyperParam,
        state: State,
        sample_fns: list[CompleteSampleFn],
        surrogate_models: list[compose.Pipeline],
    ) -> RuleClass | None:
        """Search the valid candidate with highest coverage.
        Return None if no valid candidates found."""

        best_cand: RuleClass | None
        best_cand = None

        n_features = state["n_features"]

        # t --- a tuple in best candidates
        # i --- an INDEX to the tuple t
        for i, t in b_best_cands:
            # I can choose at most (beam_size - 1) tuples at each step,
            # and there are at most n_feature steps
            beta = np.log(
                1.0
                / (
                    hyper_param.delta
                    / (1 + (hyper_param.beam_size - 1) * n_features)
                )
            )

            # Update confidence interval and coverage of the tuple t.
            mean, lb, ub = NewLimeBaseBeam.update_confidence_bound(
                t, beta, state
            )
            coverage = state["t_coverage"][t]

            # Judge whether the tuple t is an anchor or not.
            while (
                mean >= hyper_param.desired_confidence
                and lb
                < hyper_param.desired_confidence - hyper_param.epsilon_stop
            ) or (
                mean < hyper_param.desired_confidence
                and ub
                >= hyper_param.desired_confidence + hyper_param.epsilon_stop
            ):
                sample_fns[i](hyper_param.batch_size)
                mean, lb, ub = NewLimeBaseBeam.update_confidence_bound(
                    t, beta, state
                )
            ##

            # If the tuple t is the anchor with the provisionally best
            # coverage, update 'best_tuple' and 'best_model'.
            if (
                mean >= hyper_param.desired_confidence
                and lb
                > hyper_param.desired_confidence - hyper_param.epsilon_stop
            ):
                if best_cand is None or coverage > best_cand.coverage:
                    best_cand = RuleClass(
                        t, copy.deepcopy(surrogate_models[i]), None, coverage
                    )
                    # if best_cand.coverage == 1 or stop_on_first:
                    #     stop_this = True
            ##
        ##
        return best_cand

    ##

    @staticmethod
    def init_state(sample_fn: SampleFn, hyper_param: HyperParam) -> State:
        """Initialize state before starting beam search"""

        sample: Sample
        sample = sample_fn(
            [], hyper_param.coverage_samples_num, False, None, True
        )

        # data for calculating coverage of the rules
        coverage_data: np.ndarray
        coverage_data = sample.data

        prealloc_size = hyper_param.batch_size * 10000
        current_idx = 0
        data = np.zeros(
            (prealloc_size, sample.raw_data.shape[1]), coverage_data.dtype
        )
        raw_data = np.zeros(
            (prealloc_size, sample.raw_data.shape[1]), sample.raw_data.dtype
        )
        labels = np.zeros(prealloc_size, sample.labels.dtype)

        n_features = data.shape[1]

        # dictionary of set of indexes to the samples generated under the rule
        t_idx: dict[Rule, set]
        t_idx = collections.defaultdict(set)

        # dictionary of the number of the samples generated under the rule
        t_nsamples: dict[Rule, int]
        t_nsamples = collections.defaultdict(int)

        # dictionary of the number of the positive samples generated under the
        # rule
        t_positives: dict[Rule, int]
        t_positives = collections.defaultdict(int)

        # dictionary of set of indexes to the coverage samples generated under
        # the rule
        t_coverage_idx: dict[Rule, set]
        t_coverage_idx = collections.defaultdict(set)

        # dictionary of the coverages of the rules
        t_coverage: dict[Rule, float]
        t_coverage = collections.defaultdict(float)

        return {
            "t_idx": t_idx,
            "t_nsamples": t_nsamples,
            "t_positives": t_positives,
            "data": data,
            "prealloc_size": prealloc_size,
            "raw_data": raw_data,
            "labels": labels,
            "current_idx": current_idx,
            "n_features": n_features,
            "t_coverage_idx": t_coverage_idx,
            "t_coverage": t_coverage,
            "coverage_data": coverage_data,
            "t_order": collections.defaultdict(list),
        }

    ##

    @staticmethod
    def beam_search(
        sample_fn: SampleFn, hyper_param: HyperParam
    ) -> tuple[Anchor, compose.Pipeline | None] | None:
        """Beam search for optimal rule and model"""

        state: State
        state = NewLimeBaseBeam.init_state(sample_fn, hyper_param)

        # the rule with the highest coverage of the rules with higher
        # precision than tau in the B best rules
        best_rule: RuleClass | None
        best_rule = None

        prev_best_b_cands: list[Rule]
        prev_best_b_cands = []

        current_size = 0
        n_features = state["n_features"]
        best_of_size: dict[int, list[Rule]] = {}

        while current_size <= n_features:
            # -----------------------------------------------------------------
            # Call 'GenerateCands' and get new candidate rules.
            cands: list[Rule]
            cands = AnchorBaseBeam.make_tuples(prev_best_b_cands, state)
            # -----------------------------------------------------------------

            # list of indexes of the B best rules of candidate rules
            b_best_idxes: list[int]

            # list of the B best rules
            # (thus it has the same length as chosen_tuples)
            b_best_cands: list[Rule]

            # list of methods to sample perturbed vectors under the each rule
            # in the candidate rules
            # (thus it has the same length as cands)
            sample_fns: list[CompleteSampleFn]

            # list of surrogate models learned under the each rule in the B
            # best rules
            # (thus it has the same length as cands)
            surrogate_models: list[compose.Pipeline]

            # Call 'B-BestCands' and get the best B candidate rules.
            (
                b_best_idxes,
                b_best_cands,
                sample_fns,
                surrogate_models,
            ) = NewLimeBaseBeam.b_best_cands(
                cands,
                sample_fn,
                hyper_param,
                state,
            )

            best_of_size[current_size] = b_best_cands

            # -----------------------------------------------------------------

            # the rule with the highest coverage of the rules with higher
            # precision than tau in the B best rules
            # (if it does not exist, None)
            best_cand: RuleClass | None

            # Call 'LargestValidCand' and get the candidate rule with the
            # highest coverage in the best B candidate rules
            best_cand = NewLimeBaseBeam.largest_valid_cand(
                list(zip(b_best_idxes, b_best_cands)),
                hyper_param,
                state,
                sample_fns,
                surrogate_models,
            )

            if best_cand is not None:
                best_rule = best_cand
                break

            # -----------------------------------------------------------------

            # go to next iteration
            prev_best_b_cands = best_of_size[current_size]
            current_size += 1
        ##

        if best_rule is None:
            return None

        return (
            AnchorBaseBeam.get_anchor_from_tuple(best_rule.rule, state),
            best_rule.model,
        )

    ##
