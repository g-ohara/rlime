"""This is a module for beam search of best anchor.

This module contains the class NewLimeBaseBeam, which is a class for beam
search of best anchor. It is based on the class AnchorBaseBeam in the module
anchor_base.py, which is a class for beam search of best anchor. The
difference between the two classes is that the class NewLimeBaseBeam uses
surrogate models to calculate the precision of the rules, while the class
AnchorBaseBeam uses the original model to calculate the precision of the
rules.
"""

import dataclasses
from typing import Callable

import numpy as np
from anchor.anchor_base import AnchorBaseBeam
from river import compose, linear_model, preprocessing


@dataclasses.dataclass
class Samples:
    """This is a class for perturbed vectors sampled from distribution

    Attributes
    ----------
    raw_data: np.ndarray
        The perturbed vectors sampled from distribution
    data: np.ndarray
        The boolean vectors that indicates whether the feature is same as the
        target instance or not
    rewards: np.ndarray
        The boolean data that indicates whether the surrogate model predicts
        the same label as the black box model or not
    """

    raw_data: np.ndarray
    data: np.ndarray
    rewards: np.ndarray


Rule = tuple[int, ...]
SampleFn = Callable[[int, Rule, bool, compose.Pipeline | None, bool], Samples]


class Arm:
    """This is a class for an arm in multi-armed bandit problem.

    Attributes
    ----------
    n_samples: int
        The number of samples generated under the arm
    n_rewards: int
        The number of samples that the surrogate model predicts the same label
        as the black box model
    rule: Rule
        The rule under which the perturbed vectors are sampled
    surrogate_model: Pipeline
        The surrogate model
    sample_fn: SampleFn
        The function that samples perturbed vectors under the rule
    coverage: float
        The coverage of the rule
    """

    def __init__(
        self, rule: Rule, sample_fn: SampleFn, coverage_data: np.ndarray
    ) -> None:
        """Initialize the class Arm

        Parameters
        ----------
        rule: Rule
            The rule under which the perturbed vectors are sampled
        sample_fn: SampleFn
            The function that samples perturbed vectors under the rule
        coverage_data: np.ndarray
            The data for calculating coverage of the rules
        """

        self.rule = rule
        self.surrogate_model = Arm.init_surrogate_model()
        self.sample_fn = sample_fn
        self.n_samples = 0
        self.n_rewards = 0
        covered = Arm.count_covered_samples(self.rule, coverage_data)
        self.coverage = covered / coverage_data.shape[0]

    @staticmethod
    def init_surrogate_model() -> compose.Pipeline:
        """Initialize online linear model and returns it

        Returns
        -------
        Pipeline
            The initialized model
        """
        preprocessor = preprocessing.StandardScaler()
        model = linear_model.LogisticRegression()
        pipeline = compose.Pipeline(preprocessor, model)
        return pipeline

    @staticmethod
    def count_covered_samples(rule: Rule, samples: np.ndarray) -> int:
        """Count the number of samples covered by the rule

        Parameters
        ----------
        rule: Rule
            The rule under which the perturbed vectors are sampled
        samples: np.ndarray
            The perturbed vectors

        Returns
        -------
        int
            The number of samples covered by the rule
        """
        return sum(all(sample[i] == 1 for i in rule) for sample in samples)

    def sample(self, n: int) -> int:
        """Sample perturbed vectors under the arm and update the arm

        Parameters
        ----------
        n: int
            The number of perturbed vectors sampled

        Returns
        -------
        int
            The number of perturbed vectors that the surrogate model
            predicts the same label as the black box model
        """
        samples: Samples
        samples = self.sample_fn(
            n, self.rule, True, self.surrogate_model, True
        )
        self.n_samples += n
        self.n_rewards += samples.rewards.sum()
        return int(samples.rewards.sum())


@dataclasses.dataclass
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


class NewLimeBaseBeam:
    """This is a class for beam search of best anchor."""

    @staticmethod
    def make_tuples(
        previous_best: list[Rule],
        sample_fn: SampleFn,
        n_features: int,
        coverage_data: np.ndarray,
    ) -> list[Arm]:
        """Generate candidate rules.

        Parameters
        ----------
        previous_best: list
            The list of the B best rules of the previous iteration
        sample_fn: SampleFn
            The function that samples perturbed vectors under the rule
        n_features: int
            The number of features
        coverage_data: np.ndarray
            The data for calculating coverage of the rules

        Returns
        -------
        list[Arm]
            The list of the candidate rules
        """

        # Generate new candidates.
        new_cands: list[Rule]
        if len(previous_best) == 0:
            new_cands = [()]
        else:
            set_tuples = set()
            for f in range(n_features):
                for t in previous_best:
                    new_t = tuple(sorted(set(t + (f,))))
                    if len(new_t) != len(t) + 1:
                        continue
                    set_tuples.add(new_t)
            new_cands = list(set_tuples)

        # Initialize the number of samples, the number of positive samples and
        # the coverage of the rules.
        return [Arm(t, sample_fn, coverage_data) for t in new_cands]

    @staticmethod
    def lucb(cands: list[Arm], hyper_param: HyperParam) -> list[int]:
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

        top_n = min(hyper_param.beam_size, n_cands)
        if top_n >= n_cands:
            return list(range(n_cands))

        def update_bounds(t: int) -> tuple[int, int]:
            sorted_means = np.argsort(means)
            beta = AnchorBaseBeam.compute_beta(n_cands, t, hyper_param.delta)
            j = sorted_means[-top_n:]
            not_j = sorted_means[:-top_n]
            for f in not_j:
                ub[f] = AnchorBaseBeam.dup_bernoulli(
                    means[f], beta / cands[f].n_samples
                )
            for f in j:
                lb[f] = AnchorBaseBeam.dlow_bernoulli(
                    means[f], beta / cands[f].n_samples
                )
            ut = not_j[np.argmax(ub[not_j])]
            lt = j[np.argmin(lb[j])]
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
        return list(sorted_means[-top_n:])  # return top_n best indexes

    @staticmethod
    def b_best_cands(cands: list[Arm], hyper_param: HyperParam) -> list[Arm]:
        """Search for B candidates with highest precision

        Parameters
        ----------
        cands: list[Arm]
            The list of arms
        hyper_param: HyperParam
            The hyper parameters for beam search

        Returns
        -------
        list[Arm]
            The list of the B best arms
        """

        # # Learn surrogate models under each rule with initial samples without
        # # updating the state
        # for fn in sample_fns:
        #     fn(hyper_param.init_sample_num, False)

        # list of the indexes to B candidate rules with the highest precision
        b_best_idxes: list[int]
        b_best_idxes = list(NewLimeBaseBeam.lucb(cands, hyper_param))

        # list of the B candidate rules with the highest precision
        # thus it has the same length as b_best_idxes
        b_best_cands: list[Arm]
        b_best_cands = [cands[i] for i in b_best_idxes]

        # Get candidates from their indexes
        return b_best_cands

    @staticmethod
    def update_confidence_bound(
        arm: Arm, beta: float
    ) -> tuple[float, float, float]:
        """Update confidence bound of the precision of the rule

        Parameters
        ----------
        arm: Arm
            The arm
        beta: float
            The beta parameter in LUCB

        Returns
        -------
        tuple
            The tuple of the mean, lower bound and upper bound of the precision
            of the rule
        """

        mean = arm.n_rewards / arm.n_samples
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / arm.n_samples)
        ub = AnchorBaseBeam.dup_bernoulli(mean, beta / arm.n_samples)

        return mean, lb, ub

    @staticmethod
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

        Returns
        -------
        Arm
            The rule with the highest coverage of the rules with higher
            precision than tau in the B best rules
        """

        best_cand: Arm | None = None

        for cand in b_best_cands:
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
            mean, lb, ub = NewLimeBaseBeam.update_confidence_bound(cand, beta)

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
                mean, lb, ub = NewLimeBaseBeam.update_confidence_bound(
                    cand, beta
                )

            # If the tuple t is the anchor with the provisionally best
            # coverage, update 'best_tuple' and 'best_model'.
            if (
                mean >= hyper_param.tau
                and lb > hyper_param.tau - hyper_param.epsilon_stop
            ):
                if best_cand is None or cand.coverage > best_cand.coverage:
                    best_cand = cand

        return best_cand

    @staticmethod
    def beam_search(
        sample_fn: SampleFn, hyper_param: HyperParam
    ) -> Arm | None:
        """Beam search for optimal rule and model

        Parameters
        ----------
        sample_fn: Callable
            The function that samples perturbed vectors
        hyper_param: HyperParam
            The hyper parameters for beam search of best anchor

        Returns
        -------
        Arm | None
            The rule with the highest coverage of the rules with higher
            precision than tau in the B best rules
        """

        # Generate initial samples to calculate coverage of the rules
        samples: Samples
        samples = sample_fn(
            hyper_param.coverage_samples_num, (), False, None, False
        )
        coverage_data: np.ndarray
        coverage_data = samples.data
        n_features = coverage_data.shape[1]

        # the rule with the highest coverage of the rules with higher
        # precision than tau in the B best rules
        best_arm: Arm | None = None

        prev_best_b_cands: list[Arm]
        prev_best_b_cands = []

        current_size = 0

        best_of_size: dict[int, list[Arm]] = {}

        # Set maximum length of the rule
        if hyper_param.max_rule_length is None:
            hyper_param.max_rule_length = n_features

        while current_size < hyper_param.max_rule_length:
            # -----------------------------------------------------------------
            # Call 'GenerateCands' and get new candidate rules.
            arms = NewLimeBaseBeam.make_tuples(
                [cand.rule for cand in prev_best_b_cands],
                sample_fn,
                n_features,
                coverage_data,
            )
            # -----------------------------------------------------------------

            # list of the B best rules
            # (thus it has the same length as chosen_tuples)
            b_best_cands: list[Arm]

            # Call 'B-BestCands' and get the best B candidate rules.
            b_best_cands = NewLimeBaseBeam.b_best_cands(arms, hyper_param)

            best_of_size[current_size] = b_best_cands

            # -----------------------------------------------------------------

            # the rule with the highest coverage of the rules with higher
            # precision than tau in the B best rules
            # (if it does not exist, None)
            best_cand: Arm | None

            # Call 'LargestValidCand' and get the candidate rule with the
            # highest coverage in the best B candidate rules
            best_cand = NewLimeBaseBeam.largest_valid_cand(
                b_best_cands, hyper_param, n_features
            )

            if best_cand is not None:
                best_arm = best_cand
                break

            # -----------------------------------------------------------------

            # go to next iteration
            prev_best_b_cands = best_of_size[current_size]
            current_size += 1

        # end while

        return best_arm
