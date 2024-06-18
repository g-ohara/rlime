"""This is a module for beam search of best anchor.

This module contains the class NewLimeBaseBeam, which is a class for beam
search of best anchor. It is based on the class AnchorBaseBeam in the module
anchor_base.py, which is a class for beam search of best anchor. The
difference between the two classes is that the class NewLimeBaseBeam uses
surrogate models to calculate the precision of the rules, while the class
AnchorBaseBeam uses the original model to calculate the precision of the
rules.
"""

from dataclasses import dataclass

import numpy as np
from anchor.anchor_base import AnchorBaseBeam

from .arm import Arm
from .rlime_types import Classifier, Dataset, IntArray, Rule
from .sampler import Sampler


@dataclass
class HyperParam:  # pylint: disable=too-many-instance-attributes
    """Hyper parameters for beam search of best anchor

    Attributes
    ----------
    tau: float
        The threshold of precision of the rules
    delta: float
        The critical ratio for LUCB
    epsilon: float
        The allowable error for LUCB
    epsilon_stop: float
        The allowable error for searching valid rules in largest_valid_cand
    beam_size: int
        The number of candidate rules to be considered at each step
    batch_size: int
        The number of perturbed vectors generated at each step
    init_sample_num: int
        The number of initial samples for LUCB
    coverage_samples_num: int
        The number of samples generated for calculating coverage of the rules
    max_rule_length: int | None
        The maximum length of the rule (if None, it is set to the number of
        features)
    """

    tau: float = 0.80
    delta: float = 0.05
    epsilon: float = 0.05
    epsilon_stop: float = 0.05
    beam_size: int = 10
    batch_size: int = 10
    init_sample_num: int = 1000
    coverage_samples_num: int = 10000
    max_rule_length: int | None = None


def make_tuples(
    previous_best: list[Rule],
    sampler: Sampler,
    n_features: int,
    coverage_data: IntArray,
) -> list[Arm]:
    """Generate candidate rules.

    Parameters
    ----------
    previous_best: list
        The list of the B best rules of the previous iteration
    sampler: Sampler
        The sampler for sampling perturbed vectors
    n_features: int
        The number of features
    coverage_data: np.ndarray
        The data for calculating coverage of the rules

    Returns
    -------
    list[Arm]
        The list of the candidate rules
    """

    def add_features(t: Rule) -> list[Rule]:
        """Add all possible predicates to the rule

        Parameters
        ----------
        t: Rule
            The rule

        Returns
        -------
        list[Rule]
            The list of the rules with all possible predicates added to
            the rule
        """

        def normalize(t: Rule) -> Rule:
            return tuple(sorted(set(t)))

        all_features = set(range(n_features))
        return [normalize(t + (f,)) for f in all_features if f not in t]

    # Generate new candidates by adding all the possible predicates to the
    # B best rules of the previous iteration. If it is the first iteration,
    # the new candidates are the empty rule.
    new_cands: set[Rule]
    if len(previous_best) == 0:
        new_cands = {()}
    else:
        new_cands = set(sum([add_features(t) for t in previous_best], []))

    # Initialize the number of samples, the number of positive samples and
    # the coverage of the rules.
    return [Arm(t, sampler, coverage_data) for t in new_cands]


def lucb(cands: list[Arm], hyper_param: HyperParam) -> list[Arm]:
    """Search for B candidates with highest precision using LUCB algorithm

    Parameters
    ----------
    cands: list[Arm]
        The list of arms
    hyper_param: HyperParam
        The hyper parameters for beam search

    Returns
    -------
    list[int]
        The list of indexes of the B best arms
    """

    n_cands = len(cands)
    means = [0.5] * n_cands
    ub = np.array([1.0] * n_cands)
    lb = np.array([0.0] * n_cands)
    for cand in cands:
        cand.sample(hyper_param.init_sample_num)

    # The number of candidates to be considered at each step
    top_n = min(hyper_param.beam_size, n_cands)
    if top_n >= n_cands:
        return cands

    def update_bounds(t: int) -> tuple[int, int]:
        """Update the upper bound and the lower bound of the precision of the
        rules and return the indexes of the arms with the highest upper bound
        and the lowest lower bound

        Parameters
        ----------
        t: int
            The number of samples generated so far

        Returns
        -------
        tuple[int, int]
            The indexes of the arms with the highest upper bound and the
            lowest lower bound
        """

        sorted_means = np.argsort(means)
        beta = AnchorBaseBeam.compute_beta(n_cands, t, hyper_param.delta)
        top_n_means = sorted_means[-top_n:]
        not_j = sorted_means[:-top_n]
        ub[not_j] = [
            AnchorBaseBeam.dup_bernoulli(means[f], beta / cands[f].n_samples)
            for f in not_j
        ]
        lb[top_n_means] = [
            AnchorBaseBeam.dlow_bernoulli(means[f], beta / cands[f].n_samples)
            for f in top_n_means
        ]
        ut = not_j[np.argmax(ub[not_j])]
        lt = top_n_means[np.argmin(lb[top_n_means])]
        return ut, lt

    t = 1
    ut, lt = update_bounds(t)
    while ub[ut] - lb[lt] > hyper_param.epsilon:
        ut_arm = cands[ut]
        ut_arm.sample(hyper_param.batch_size)
        means[ut] = ut_arm.n_rewards / ut_arm.n_samples
        lt_arm = cands[lt]
        lt_arm.sample(hyper_param.batch_size)
        means[lt] = lt_arm.n_rewards / lt_arm.n_samples
        t += 1
        ut, lt = update_bounds(t)
    sorted_means = np.argsort(means)
    return [cands[i] for i in sorted_means[-top_n:]]


def largest_valid_cand(
    b_best_cands: list[Arm],
    hyper_param: HyperParam,
    n_features: int,
) -> Arm | None:
    """Search the valid candidate with highest coverage. Return None if no
    valid candidates found.

    Parameters
    ----------
    b_best_cands: list
        The list of the B best rules
    hyper_param: HyperParam
        The hyper parameters for beam search of best anchor
    n_features: int
        The number of features

    Returns
    -------
    Arm
        The rule with the highest coverage of the rules with higher
        precision than tau in the B best rules
    """

    def update_confidence_bound(
        arm: Arm, beta: float
    ) -> tuple[float, float, float]:
        """Update confidence bound of the precision of the rule"""
        mean = arm.n_rewards / arm.n_samples
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / arm.n_samples)
        ub = AnchorBaseBeam.dup_bernoulli(mean, beta / arm.n_samples)
        return mean, lb, ub

    delta = hyper_param.delta
    beam_size = hyper_param.beam_size
    beta = np.log(1.0 / (delta / (1 + (beam_size - 1) * n_features)))

    best_cand: Arm | None = None
    for cand in b_best_cands:
        # Update confidence interval and coverage of the tuple t.
        mean, lb, ub = update_confidence_bound(cand, beta)

        # Judge whether the rule is valid or not.
        while (
            mean >= hyper_param.tau
            and lb < hyper_param.tau - hyper_param.epsilon_stop
            or mean < hyper_param.tau
            and ub >= hyper_param.tau + hyper_param.epsilon_stop
        ):
            # If it is not clear whether the rule is valid or not,
            # sample more perturbed vectors and update the confidence
            # interval and coverage of the rule.
            cand.sample(hyper_param.batch_size)
            mean, lb, ub = update_confidence_bound(cand, beta)

        # If the tuple t is the valid rule with the provisionally best
        # coverage, update the best rule.
        if (
            mean >= hyper_param.tau
            and lb > hyper_param.tau - hyper_param.epsilon_stop
        ):
            if best_cand is None or cand.coverage > best_cand.coverage:
                best_cand = cand

    return best_cand


def beam_search(sampler: Sampler, hyper_param: HyperParam) -> Arm | None:
    """Beam search for optimal rule and model

    Parameters
    ----------
    sampler: Sampler
        The sampler for sampling perturbed vectors
    hyper_param: HyperParam
        The hyper parameters for beam search of best anchor

    Returns
    -------
    Arm | None
        The rule with the highest coverage of the rules with higher
        precision than tau in the B best rules
        (if it does not exist, None)
    """

    # Generate initial samples to calculate coverage of the rules
    coverage_data, _ = sampler.sample(hyper_param.coverage_samples_num, ())
    n_features = coverage_data.shape[1]

    # Set maximum length of the rule
    max_rule_length = hyper_param.max_rule_length
    if max_rule_length is None:
        max_rule_length = n_features

    b_best_cands: list[Arm] = []

    for _ in range(max_rule_length):
        # Call 'GenerateCands' and get new candidate rules by adding all the
        # possible predicates to the B best rules of the previous iteration.
        arms = make_tuples(
            [cand.rule for cand in b_best_cands],
            sampler,
            n_features,
            coverage_data,
        )

        # Call 'B-BestCands' and get the B candidate rules with the highest
        # precision.
        b_best_cands = lucb(arms, hyper_param)

        # Call 'LargestValidCand' and get the candidate rule with the highest
        # coverage of the rules with higher precision than tau in the B best
        # rules. If no such rule exists, return None.
        best_cand = largest_valid_cand(b_best_cands, hyper_param, n_features)

        # If the candidate rule is found, return the rule.
        if best_cand is not None:
            return best_cand

    return None


def explain_instance(
    trg: IntArray,
    dataset: Dataset,
    classifier_fn: Classifier,
    hyper_param: HyperParam,
) -> tuple[list[str], Arm] | None:
    """Generate NewLIME explanation for given classifier on neighborhood
    of given data point.

    Parameters
    ----------
    trg: IntArray
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
        trg, dataset.data, classifier_fn, dataset.categorical_names
    )
    arm = beam_search(sampler, hyper_param)
    if arm is None:
        return None

    # Get names from the rule
    names = []
    for i in arm.rule:
        if i in dataset.ordinal_features:
            name = dataset.categorical_names[i][int(trg[i])]
        else:
            name = (
                dataset.feature_names[i]
                + " = "
                + dataset.categorical_names[i][int(trg[i])]
            )
        names.append(name)

    return names, arm
