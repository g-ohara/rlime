"""This module provides a class for sampling data points from the conditional
multivariate Gaussian distribution. The mean and covariance matrix of the
distribution are computed from the training data and the given conditions.
"""

import numpy as np

from .rlime_types import Classifier, FloatArray, IntArray, Rule


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

        Returns:
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

        Returns:
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

        Returns:
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
