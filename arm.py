"""This module contains the class for an arm in multi-armed bandit problem"""

import pandas as pd
from river import compose, linear_model, preprocessing

from .rlime_types import IntArray, Rule
from .sampler import Sampler


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
        covered = Arm.count_covered_samples(
            self.rule, sampler.trg, coverage_data
        )
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
    def count_covered_samples(
        rule: Rule, trg: IntArray, samples: IntArray
    ) -> int:
        """Count the number of samples covered by the rule

        Parameters
        ----------
        rule: Rule
            The rule under which the perturbed vectors are sampled
        trg: IntArray
            The target instance
        samples: np.ndarray
            The perturbed vectors

        Returns
        -------
        int
            The number of samples covered by the rule
        """
        return sum(
            all(sample[i] == trg[i] for i in rule) for sample in samples
        )

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
