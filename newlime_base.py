"""Base anchor functions"""

from __future__ import print_function

import copy
import collections
from typing import Callable
from typing import Any

from river import compose
from river import linear_model
from river import preprocessing

import numpy as np
from anchor import anchor_base

Predicate = tuple[int, str, int]
Rule = tuple[int, ...]
Mapping = dict[int, Predicate]
Classifier = Callable[[np.ndarray], np.ndarray]
SampleFn = Callable[
    [list[int], int, bool, compose.Pipeline | None, bool],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]
CompleteSampleFn = Callable[[int], float]
State = dict[str, Any]
Anchor = dict[str, Any]


class Arm:
    def __init__(self, rule: Rule, sample_fn: SampleFn) -> None:
        self.rule = rule
        self.model = compose.Pipeline(
            preprocessing.StandardScaler(), linear_model.LogisticRegression()
        )
        self.sample_fn = sample_fn

    def get_reward(self, sample_num: int, state: State) -> float:
        return AnchorBaseBeam.complete_sample_fn(
            self.rule, sample_num, self.model, self.sample_fn, state
        )


class AnchorBaseBeam(anchor_base.AnchorBaseBeam):
    @staticmethod
    def complete_sample_fn(
        t: Rule,
        n: int,
        model: compose.Pipeline,
        sample_fn: SampleFn,
        state: State,
    ) -> float:
        # *****************************************************************
        # Take a sample satisfying the tuple t.
        labels: np.ndarray
        raw_data, data, labels = sample_fn(list(t), n, True, model, True)
        # *****************************************************************

        current_idx = state["current_idx"]
        # idxs = range(state['data'].shape[0], state['data'].shape[0] + n)
        idxs: range = range(current_idx, current_idx + n)

        if "<U" in str(raw_data.dtype):
            # String types: make sure both string types are of maximum length
            # to avoid string truncation. E.g., '<U308', '<U290' -> '<U308'
            max_dtype = max(str(state["raw_data"].dtype), str(raw_data.dtype))
            state["raw_data"] = state["raw_data"].astype(max_dtype)
            raw_data = raw_data.astype(max_dtype)

        state["t_idx"][t].update(idxs)
        state["t_nsamples"][t] += n
        state["t_positives"][t] += labels.sum()
        state["data"][idxs] = data
        state["raw_data"][idxs] = raw_data
        if len(labels) > 0:
            state["labels"][idxs] = labels
        state["current_idx"] += n
        if state["current_idx"] >= state["data"].shape[0] - max(1000, n):
            prealloc_size = state["prealloc_size"]
            current_idx = data.shape[0]
            state["data"] = np.vstack(
                (state["data"], np.zeros((prealloc_size, data.shape[1]), data.dtype))
            )
            state["raw_data"] = np.vstack(
                (
                    state["raw_data"],
                    np.zeros((prealloc_size, raw_data.shape[1]), raw_data.dtype),
                )
            )
            state["labels"] = np.hstack(
                (state["labels"], np.zeros(prealloc_size, labels.dtype))
            )
        # This can be really slow
        # state['data'] = np.vstack((state['data'], data))
        # state['raw_data'] = np.vstack((state['raw_data'], raw_data))
        # state['labels'] = np.hstack((state['labels'], labels))
        ret: float = labels.sum()
        return ret

    ## end function

    @staticmethod
    def init_surrogate_models(num_models: int) -> list[compose.Pipeline]:
        models: list[compose.Pipeline] = []
        for i in range(num_models):
            preprocessor = preprocessing.StandardScaler()
            model = linear_model.LogisticRegression()
            pipeline = compose.Pipeline(preprocessor, model)
            models.append(pipeline)
        return models

    ## end function

    @staticmethod
    def get_sample_fns(
        sample_fn: SampleFn,
        tuples: list[Rule],
        state: State,
        surrogate_models: list[compose.Pipeline],
    ) -> list[CompleteSampleFn]:
        # each sample fn returns number of instances that the surrogate model
        # classified correctly
        sample_fns: list[CompleteSampleFn] = []

        for t, m in zip(tuples, surrogate_models):
            fn = lambda n, t=t, m=m, fn=sample_fn, s=state: AnchorBaseBeam.complete_sample_fn(
                t, n, m, fn, s
            )
            sample_fns.append(fn)

        return sample_fns

    ## end function

    @staticmethod
    def generate_cands(
        previous_bests: list[Rule], best_coverage: float, state: State
    ) -> list[Rule]:
        # list of the candidate rules generated from the previous B best rules
        cands: list[Rule]
        cands = AnchorBaseBeam.make_tuples(previous_bests, state)

        # list of the candidate rules with higher coverage than that of the
        # provisionally best rule already found
        good_cands: list[Rule]
        good_cands = [x for x in cands if state["t_coverage"][x] > best_coverage]

        return good_cands

    ##

    @staticmethod
    def b_best_cands(
        tuples: list[Rule],
        sample_fn: SampleFn,
        beam_size: int,
        epsilon: float,
        delta: float,
        batch_size: int,
        top_n: int,
        state: State,
        verbose: bool,
        verbose_every: int,
    ) -> tuple[list[int], list[Rule], list[CompleteSampleFn], list[compose.Pipeline]]:
        # list of the surrogate models under each rule
        # thus it has the same length as 'tuples'
        surrogate_models: list[compose.Pipeline]

        # Initialize surrogate models
        surrogate_models = AnchorBaseBeam.init_surrogate_models(len(tuples))

        # list of the functions to sample perturbed vectors under each rule
        # thus it has the same length as 'tuples'
        sample_fns: list[CompleteSampleFn]

        # Get sampling functions
        sample_fns = AnchorBaseBeam.get_sample_fns(
            sample_fn, tuples, state, surrogate_models
        )

        initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)

        chosen_tuples = AnchorBaseBeam.lucb(
            sample_fns,
            initial_stats,
            epsilon,
            delta,
            batch_size,
            top_n,
            verbose,
            verbose_every,
        )

        # Get candidates from their indexes
        return (
            list(chosen_tuples),
            [tuples[x] for x in chosen_tuples],
            sample_fns,
            surrogate_models,
        )

    ##

    @staticmethod
    def update_confidence_bound(
        rule: Rule, batch_size: int, beta: float, state: State
    ) -> tuple[float, float, float]:
        mean = state["t_positives"][rule] / state["t_nsamples"][rule]
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / state["t_nsamples"][rule])
        ub = AnchorBaseBeam.dup_bernoulli(mean, beta / state["t_nsamples"][rule])

        return mean, lb, ub

    ##

    @staticmethod
    def largest_valid_cand(
        chosen_tuples: list[int],
        tuples: list[Rule],
        delta: float,
        beam_size: int,
        n_features: int,
        state: State,
        verbose: bool,
        desired_confidence: float,
        epsilon_stop: float,
        sample_fns: list[CompleteSampleFn],
        batch_size: int,
        surrogate_models: list[compose.Pipeline],
        stop_on_first: bool,
    ) -> tuple[Rule, compose.Pipeline | None, float, bool]:
        best_cand: Rule
        best_cand_model: compose.Pipeline | None
        best_cand_coverage: float
        stop_this: bool

        best_cand = ()
        best_cand_model = None
        best_cand_coverage = 0.0
        stop_this = False

        # t --- a tuple in best candidates
        # i --- an INDEX to the tuple t
        for i, t in list(zip(chosen_tuples, tuples)):
            # I can choose at most (beam_size - 1) tuples at each step,
            # and there are at most n_feature steps
            beta = np.log(1.0 / (delta / (1 + (beam_size - 1) * n_features)))

            # Update confidence interval and coverage of the tuple t.
            mean, lb, ub = AnchorBaseBeam.update_confidence_bound(
                t, batch_size, beta, state
            )
            coverage = state["t_coverage"][t]
            if verbose:
                print(i, mean, lb, ub)

            # Judge whether the tuple t is an anchor or not.
            while (
                mean >= desired_confidence and lb < desired_confidence - epsilon_stop
            ) or (
                mean < desired_confidence and ub >= desired_confidence + epsilon_stop
            ):
                sample_fns[i](batch_size)
                mean, lb, ub = AnchorBaseBeam.update_confidence_bound(
                    t, batch_size, beta, state
                )
            ## end while

            # If the tuple t is the anchor with the provisionally best
            # coverage, update 'best_tuple' and 'best_model'.
            if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:
                if coverage > best_cand_coverage:
                    best_cand_coverage = coverage
                    best_cand = t
                    best_cand_model = copy.deepcopy(surrogate_models[i])
                    if best_cand_coverage == 1 or stop_on_first:
                        stop_this = True
            ## end if
        ## end for
        return best_cand, best_cand_model, best_cand_coverage, stop_this

    ##

    @staticmethod
    def anchor_beam(
        sample_fn: SampleFn,
        delta: float = 0.05,
        epsilon: float = 0.1,
        batch_size: int = 10,
        desired_confidence: float = 1.0,
        beam_size: int = 1,
        verbose: bool = False,
        epsilon_stop: float = 0.05,
        max_anchor_size: int | None = None,
        verbose_every: int = 1,
        stop_on_first: bool = False,
        coverage_samples: int = 10000,
        my_verbose: bool = False,
    ) -> tuple[Anchor, compose.Pipeline | None]:
        raw_data, coverage_data, labels = sample_fn(
            [], coverage_samples, False, None, True
        )

        beta = np.log(1.0 / delta)

        prealloc_size = batch_size * 10000
        current_idx = 0
        data = np.zeros((prealloc_size, raw_data.shape[1]), coverage_data.dtype)
        raw_data = np.zeros((prealloc_size, raw_data.shape[1]), raw_data.dtype)
        labels = np.zeros(prealloc_size, labels.dtype)

        n_features = data.shape[1]
        state: State = {
            "t_idx": collections.defaultdict(lambda: set()),
            "t_nsamples": collections.defaultdict(lambda: 0.0),
            "t_positives": collections.defaultdict(lambda: 0.0),
            "data": data,
            "prealloc_size": prealloc_size,
            "raw_data": raw_data,
            "labels": labels,
            "current_idx": current_idx,
            "n_features": n_features,
            "t_coverage_idx": collections.defaultdict(lambda: set()),
            "t_coverage": collections.defaultdict(lambda: 0.0),
            "coverage_data": coverage_data,
            "t_order": collections.defaultdict(lambda: list()),
        }

        current_size = 0
        best_of_size: dict[int, list[Rule]] = {}

        # the rule with the highest coverage of the rules with higher
        # precision than tau in the B best rules
        best_tuple: Rule

        # the surrogate model learned under best_tuple
        best_model: compose.Pipeline | None

        # coverage of best_tuple
        best_coverage: float

        best_tuple = ()
        best_model = None
        best_coverage = -1.0

        if max_anchor_size is None:
            max_anchor_size = n_features

        prev_best_b_cands: list[Rule]
        prev_best_b_cands = []

        while current_size <= max_anchor_size:
            # newly generated candidate rules
            cands: list[Rule]

            # Call 'GenerateCands' and get new candidate rules.
            cands = AnchorBaseBeam.generate_cands(
                prev_best_b_cands, best_coverage, state
            )

            # Exit if no candidate rules are generated
            if len(cands) == 0:
                if my_verbose:
                    print("Cannot generate new candidate rules")
                break

            # -----------------------------------------------------------------

            # list of indexes of the B best rules of candidate rules
            chosen_tuples: list[int]

            # list of the B best rules
            # (thus it has the same length as chosen_tuples)
            chosen_rules: list[Rule]

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
                chosen_tuples,
                chosen_rules,
                sample_fns,
                surrogate_models,
            ) = AnchorBaseBeam.b_best_cands(
                cands,
                sample_fn,
                beam_size,
                epsilon,
                delta,
                batch_size,
                min(beam_size, len(cands)),
                state,
                verbose,
                verbose_every,
            )

            best_of_size[current_size] = chosen_rules

            if verbose:
                print("Best of size ", current_size, ":")

            # -----------------------------------------------------------------

            # the rule with the highest coverage of the rules with higher
            # precision than tau in the B best rules
            best_cand: Rule

            # the surrogate model learned under best_tuple
            best_cand_model: compose.Pipeline | None

            # coverage of best_tuple
            best_cand_coverage: float

            # Call 'LargestValidCand' and get the candidate rule with the
            # highest coverage in the best B candidate rules
            (
                best_cand,
                best_cand_model,
                best_cand_coverage,
                stop_this,
            ) = AnchorBaseBeam.largest_valid_cand(
                chosen_tuples,
                cands,
                delta,
                beam_size,
                n_features,
                state,
                verbose,
                desired_confidence,
                epsilon_stop,
                sample_fns,
                batch_size,
                surrogate_models,
                stop_on_first,
            )

            if best_cand_coverage > best_coverage:
                best_tuple = copy.deepcopy(best_cand)
                best_model = copy.deepcopy(best_cand_model)
                best_coverage = best_cand_coverage

            if stop_this:
                if my_verbose:
                    print("Stop This!")
                break

            # -----------------------------------------------------------------

            # go to next iteration
            prev_best_b_cands = best_of_size[current_size]
            current_size += 1

        ## end while

        if best_tuple == ():
            # Could not find an anchor, will now choose the highest precision
            # amongst the top K from every round
            if verbose:
                print("Could not find an anchor, now doing best of each size")

            # list of the rules with the highest precision in each round
            tuples: list[Rule]
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])

            (
                chosen_tuples,
                _,
                sample_fns,
                surrogate_models,
            ) = AnchorBaseBeam.b_best_cands(
                tuples,
                sample_fn,
                beam_size,
                epsilon,
                delta,
                batch_size,
                1,
                state,
                verbose,
                verbose_every,
            )
            best_tuple = tuples[chosen_tuples[0]]
            best_model = surrogate_models[chosen_tuples[0]]
        ## end if

        # return best_tuple, state

        best_anchor = AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)
        if my_verbose:
            print("202310031631")
        return best_anchor, best_model

    ## end function
