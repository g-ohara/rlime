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
import numpy.typing as npt
import pandas as pd
from anchor.anchor_base import AnchorBaseBeam
from river import compose, linear_model, preprocessing

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
Rule = tuple[int, ...]
Classifier = Callable[[IntArray | FloatArray], IntArray]


class Sampler:
    """This class provides sampling functions for NewLIME. The sample is taken
    from the conditional multivariate Gaussian distribution. The mean and
    covariance matrix of the distribution are computed from the training data
    and the given conditions.
    """

    def __init__(
        self,
        trg: IntArray,
        train: IntArray,
        black_box_model: Classifier,
        categorical_names: dict[int, list[str]],
    ):
        """Initialize Sampler.

        Parameters
        ----------
        trg : np.ndarray
            Target instance.
        train : np.ndarray
            Training data for computing mean and covariance matrix of the
            conditional multivariate Gaussian distribution.
        black_box_model : Classifier
            Black box model.
        categorical_names : dict[int, list[str]]
            Dictionary of categorical feature names and possible values.
        """

        self.trg = trg
        self.black_box_model = black_box_model
        self.rng = np.random.default_rng()
        self.category_counts = {
            i: len(v) for i, v in categorical_names.items()
        }

        self.params: dict[Rule, tuple[FloatArray, FloatArray]]
        emp_mean = np.mean(train, axis=0)
        emp_cov = np.cov(train, rowvar=False)
        self.params = {(): (emp_mean, emp_cov)}

    def get_params(self, rule: Rule) -> tuple[FloatArray, FloatArray]:
        """Compute mean and covariance matrix of conditional multivariate
        Gaussian distribution.

        Parameters
        ----------
        rule : Rule
            Target rule. The rule is represented as a tuple of feature indices.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Mean and covariance matrix of conditional multivariate Gaussian
            distribution.
        """

        # Get indices of features in the rule and not in the rule
        fix_idx = list(rule)
        var_idx = list(filter(lambda i: i not in rule, range(len(self.trg))))

        # Compute mean and covariance matrix of conditional multivariate
        # Gaussian distribution
        emp_mean, emp_cov = self.params[()]
        var_mean = emp_mean[var_idx] + np.dot(
            np.dot(
                emp_cov[var_idx][:, fix_idx],
                np.linalg.inv(emp_cov[fix_idx][:, fix_idx]),
            ),
            (self.trg[fix_idx] - emp_mean[fix_idx]),
        )
        var_cov = emp_cov[var_idx][:, var_idx] - np.dot(
            np.dot(
                emp_cov[var_idx][:, fix_idx],
                np.linalg.inv(emp_cov[fix_idx][:, fix_idx]),
            ),
            emp_cov[fix_idx][:, var_idx],
        )

        # Compute conditional mean and covariance matrix
        cond_mean: FloatArray = np.zeros_like(self.trg)
        cond_mean[fix_idx] = self.trg[fix_idx]
        cond_mean[var_idx] = var_mean
        cond_cov = np.zeros_like(emp_cov)
        cond_cov[np.ix_(var_idx, var_idx)] = var_cov

        return cond_mean, cond_cov

    def discretize(self, data: FloatArray) -> IntArray:
        """Discretize data points.

        Parameters
        ----------
        data : np.ndarray
            Data points to be discretized.

        Returns
        -------
        np.ndarray
            Discretized data points.
        """

        # Round continuous features and clip categorical features
        d_data: IntArray = np.zeros_like(data)
        for f, count in self.category_counts.items():
            d_data[:, f] = np.round(np.clip(data[:, f], 0, count - 1))

        return d_data

    def sample(
        self, num_samples: int, rule: Rule | None = None
    ) -> tuple[IntArray, IntArray]:
        """Sample data points from conditional multivariate Gaussian
        distribution.

        Parameters
        ----------
        num_samples : int
            The number of returned sample.
        rule : Rule, optional (default=None)
            Target rule. The rule is represented as a tuple of feature indices.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The perturbed vectors sampled from distribution and the boolean
            vectors that indicates whether the feature is same as the target
            instance or not.
        """

        if rule is None:
            rule = ()

        if rule not in self.params:
            self.params[rule] = self.get_params(rule)

        # Sample from conditional multivariate Gaussian distribution
        mean, cov = self.params[rule]
        sampled_data = self.rng.multivariate_normal(mean, cov, num_samples)

        # Discretize sampled data points
        d_full_data = self.discretize(sampled_data)

        # Predict labels of sampled data points
        psuedo_labels = self.black_box_model(d_full_data)

        return np.array(d_full_data), np.array(psuedo_labels)


class Arm:
    """This is a class for an arm in multi-armed bandit problem.

    Attributes
    ----------
    rule: Rule
        The rule under which the perturbed vectors are sampled
    surrogate_model: Pipeline
        The surrogate model
    n_samples: int
        The number of samples generated under the arm
    n_rewards: int
        The number of samples that the surrogate model predicts the same label
        as the black box model
    coverage: float
        The coverage of the rule
    """

    def __init__(
        self, rule: Rule, sampler: Sampler, coverage_data: IntArray
    ) -> None:
        """Initialize the class Arm

        Parameters
        ----------
        rule: Rule
            The rule under which the perturbed vectors are sampled
        sampler: Sampler
            The sampler for sampling perturbed vectors
        coverage_data: np.ndarray
            The data for calculating coverage of the rules
        """

        self.rule = rule
        self.surrogate_model = Arm.init_surrogate_model()
        self.sampler = sampler
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
    def count_covered_samples(rule: Rule, samples: IntArray) -> int:
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

    def sample(self, n: int) -> None:
        """Sample perturbed vectors under the arm and update the arm

        Parameters
        ----------
        n: int
            The number of perturbed vectors sampled
        """

        raw_data, psuedo_labels = self.sampler.sample(n, self.rule)

        self.n_samples += n
        data_x: pd.DataFrame = pd.DataFrame(raw_data)
        data_y = pd.Series(psuedo_labels)
        surrogate_pred = self.surrogate_model.predict_many(data_x)
        self.n_rewards += sum((surrogate_pred == data_y).astype(int))

        self.surrogate_model.learn_many(data_x, data_y)

        # return int(samples.rewards.sum())


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
