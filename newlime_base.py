"""This is a module for beam search of best anchor.

This module contains the class NewLimeBaseBeam, which is a class for beam
search of best anchor. It is based on the class AnchorBaseBeam in the module
anchor_base.py, which is a class for beam search of best anchor. The
difference between the two classes is that the class NewLimeBaseBeam uses
surrogate models to calculate the precision of the rules, while the class
AnchorBaseBeam uses the original model to calculate the precision of the
rules.
"""

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
class Samples:
    """This is a class for perturbed vectors sampled from distribution

    Attributes:
        raw_data: The original data
        data: The discretized data
        labels: The labels of the perturbed data
    """

    raw_data: np.ndarray
    data: np.ndarray
    labels: np.ndarray


SampleFn = Callable[[Rule, int, bool, compose.Pipeline | None, bool], Samples]
CompleteSampleFn = Callable[[int], float]
State = dict[str, Any]
Anchor = dict[str, Any]


@dataclasses.dataclass
class RuleClass:
    """This is the class for a rule, model learned under that rule,
    precision of that model and coverage of that rule.

    Attributes:
        rule: The rule
        model: The model learned under that rule
        precision: The precision of that model
        coverage: The coverage of that rule
    """

    rule: Rule
    model: compose.Pipeline | None
    precision: float | None
    coverage: float


@dataclasses.dataclass
class HyperParam:  # pylint: disable=too-many-instance-attributes
    """Hyper parameters for beam search of best anchor

    Attributes:
        delta: The delta parameter in LUCB
        epsilon: The epsilon parameter in LUCB
        epsilon_stop: The epsilon_stop parameter in LUCB
        beam_size: The beam_size parameter in LUCB
        batch_size: The batch_size parameter in LUCB
        desired_confidence: The tau parameter in LUCB
        coverage_samples_num: The number of samples to calculate coverage
        max_rule_length: The maximum length of the rule
    """

    delta: float = 0.05
    epsilon: float = 0.1
    epsilon_stop: float = 0.05
    beam_size: int = 10
    batch_size: int = 10
    desired_confidence: float = 0.95  # tau
    coverage_samples_num: int = 10000
    max_rule_length: int | None = None


class NewLimeBaseBeam:
    """This is a class for beam search of best anchor.

    Attributes
    ----------
    None

    Methods
    -------
    update_state(state, rule, batch_size, sample)
        Update state after sampling
    complete_sample_fn(sample_fn, rule, n, model, state)
        Sample perturbed vectors, update state and returns the reward
    init_surrogate_models(num_models)
        Initialize online linear models and returns them
    get_sample_fns(sample_fn, tuples, surrogate_models, state)
        Return function that samples, updates model and returns sample
    get_initial_statistics(tuples)
        Returns statistics to pass to AnchorBaseBeam.lucb
    b_best_cands(cands, sample_fn, hyper_param, state)
        Search for B candidates with highest precision
    update_confidence_bound(rule, beta, state)
        Update confidence bound of the precision of the rule based on state
    largest_valid_cand(
        b_best_cands, hyper_param, state, sample_fns, surrogate_models
    )
        Search the valid candidate with highest coverage.
    init_state(sample_fn, hyper_param)
        Initialize state before starting beam search
    beam_search(sample_fn, hyper_param)
        Beam search for optimal rule and model
    """

    @staticmethod
    def update_state(
        state: State, rule: Rule, batch_size: int, samples: Samples
    ) -> None:
        """Update state after sampling

        Parameters
        ----------
        state: State
            The state of the beam search
        rule: tuple
            The rule under which the perturbed vectors are sampled
        batch_size: int
            The number of perturbed vectors sampled
        samples: Samples
            The sampled perturbed vectors
        """

        current_idx: int
        current_idx = state["current_idx"]

        idxs: range
        idxs = range(current_idx, current_idx + batch_size)

        if "<U" in str(samples.raw_data.dtype):
            # String types: make sure both string types are of maximum length
            # to avoid string truncation. E.g., '<U308', '<U290' -> '<U308'
            max_dtype = max(
                str(state["raw_data"].dtype), str(samples.raw_data.dtype)
            )
            state["raw_data"] = state["raw_data"].astype(max_dtype)
            samples.raw_data = samples.raw_data.astype(max_dtype)

        state["t_idx"][rule].update(idxs)
        state["t_nsamples"][rule] += batch_size
        state["t_positives"][rule] += samples.labels.sum()
        state["data"][idxs] = samples.data
        state["raw_data"][idxs] = samples.raw_data
        if len(samples.labels) > 0:
            state["labels"][idxs] = samples.labels
        state["current_idx"] += batch_size
        if state["current_idx"] >= state["data"].shape[0] - max(
            1000, batch_size
        ):
            prealloc_size = state["prealloc_size"]
            current_idx = samples.data.shape[0]
            state["data"] = np.vstack(
                (
                    state["data"],
                    np.zeros(
                        (prealloc_size, samples.data.shape[1]),
                        samples.data.dtype,
                    ),
                )
            )
            state["raw_data"] = np.vstack(
                (
                    state["raw_data"],
                    np.zeros(
                        (prealloc_size, samples.raw_data.shape[1]),
                        samples.raw_data.dtype,
                    ),
                )
            )
            state["labels"] = np.hstack(
                (
                    state["labels"],
                    np.zeros(prealloc_size, samples.labels.dtype),
                )
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
    ) -> int:
        """Sample perturbed vectors, update state and returns the reward

        Parameters
        ----------
        sample_fn: SampleFn
            The function that ONLY samples perturbed vectors.
            It does NOT update the surrogate model.
        rule: tuple
            The rule under which the perturbed vectors are sampled
        n: int
            The number of perturbed vectors sampled
        model: Pipeline
            The model learned under the rule
        state: dict
            The state of the beam search

        Returns
        -------
        float
            The reward
        """

        # *****************************************************************
        # Take a sample satisfying the tuple t.
        samples: Samples
        samples = sample_fn(rule, n, True, model, True)
        # *****************************************************************

        NewLimeBaseBeam.update_state(state, rule, n, samples)

        return int(samples.labels.sum())

    ##

    @staticmethod
    def init_surrogate_models(num_models: int) -> list[compose.Pipeline]:
        """Initialize online linear models and returns them

        Parameters
        ----------
        num_models: int
            The number of models to initialize

        Returns
        -------
        list
            The list of initialized models
        """
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
        """Return function that samples, updates model and returns sample

        Parameters
        ----------
        sample_fn: Callable
            The function that samples perturbed vectors, updates the
            surrogate model and returns the reward
        tuples: list
            The list of rules
        surrogate_models: list
            The list of surrogate models
        state: dict
            The state of the beam search

        Returns
        -------
        list[CompleteSampleFn]
            The list of functions that sample, update model and return sample
        """
        # each sample fn returns number of instances that the surrogate model
        # classified correctly
        sample_fns: list[CompleteSampleFn] = []

        for rule, model in zip(tuples, surrogate_models):
            partial_fn = functools.partial(
                NewLimeBaseBeam.complete_sample_fn,
                sample_fn,
                rule,
                model=model,
                state=state,
            )
            sample_fns.append(partial_fn)

        return sample_fns

    ##

    @staticmethod
    def get_initial_statistics(tuples: list[Rule]) -> dict[str, np.ndarray]:
        """Returns statistics to pass to AnchorBaseBeam.lucb

        Parameters
        ----------
        tuples: list
            The list of rules

        Returns
        -------
        dict
            The dictionary of statistics
        """
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
        """Search for B candidates with highest precision

        Parameters
        ----------
        cands: list
            The list of candidate rules
        sample_fn: Callable
            The function that samples perturbed vectors
        hyper_param: HyperParam
            The hyper parameters for beam search of best anchor
        state: dict
            The state of the beam search

        Returns
        -------
        tuple[list[int], list[Rule], list[CompleteSampleFn], list[Pipeline]]
            The tuple of indexes of the B best rules, the list of the B best
            rules, the list of functions that sample, update model and return
            sample, the list of surrogate models
        """
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

        for fn in sample_fns:
            fn(hyper_param.batch_size)

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
                min(hyper_param.beam_size, len(cands)),
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
        """Update confidence bound of the precision of the rule based on state

        Parameters
        ----------
        rule: tuple
            The rule under which the perturbed vectors are sampled
        beta: float
            The beta parameter in LUCB
        state: State
            The state of the beam search

        Returns
        -------
        tuple
            The tuple of the mean, lower bound and upper bound of the precision
            of the rule
        """

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
        """Search the valid candidate with highest coverage. Return None if no
        valid candidates found.

        Parameters
        ----------
        b_best_cands: list
            The list of the B best rules
        hyper_param: HyperParam
            The hyper parameters for beam search of best anchor
        state: State
            The state of the beam search
        sample_fns: list
            The list of functions that sample, update model and return sample
        surrogate_models: list
            The list of surrogate models

        Returns
        -------
        RuleClass
            The rule with the highest coverage of the rules with higher
            precision than tau in the B best rules
        """

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
        """Initialize state before starting beam search

        Parameters
        ----------
        sample_fn: Callable
            The function that samples perturbed vectors
        hyper_param: HyperParam
            The hyper parameters for beam search of best anchor

        Returns
        -------
        dict
            The state of the beam search
        """

        samples: Samples
        samples = sample_fn(
            (), hyper_param.coverage_samples_num, False, None, True
        )

        # data for calculating coverage of the rules
        coverage_data: np.ndarray
        coverage_data = samples.data

        prealloc_size = hyper_param.batch_size * 10000
        current_idx = 0
        data = np.zeros(
            (prealloc_size, samples.raw_data.shape[1]), coverage_data.dtype
        )
        raw_data = np.zeros(
            (prealloc_size, samples.raw_data.shape[1]), samples.raw_data.dtype
        )
        labels = np.zeros(prealloc_size, samples.labels.dtype)

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
        """Beam search for optimal rule and model

        Parameters
        ----------
        sample_fn: Callable
            The function that samples perturbed vectors
        hyper_param: HyperParam
            The hyper parameters for beam search of best anchor

        Returns
        -------
        tuple[Anchor, Pipeline | None] | None
            The tuple of the anchor and the model learned under the anchor
            (if it exists, None otherwise)
        """

        state: State
        state = NewLimeBaseBeam.init_state(sample_fn, hyper_param)

        # the rule with the highest coverage of the rules with higher
        # precision than tau in the B best rules
        best_rule: RuleClass | None
        best_rule = None

        prev_best_b_cands: list[Rule]
        prev_best_b_cands = []

        current_size = 0

        best_of_size: dict[int, list[Rule]] = {}

        # Set maximum length of the rule
        if hyper_param.max_rule_length is None:
            hyper_param.max_rule_length = state["n_features"]

        while current_size < hyper_param.max_rule_length:
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

        # end while

        if best_rule is None:
            return None

        return (
            AnchorBaseBeam.get_anchor_from_tuple(best_rule.rule, state),
            best_rule.model,
        )

    ##
