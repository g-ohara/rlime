"""This module contains the LIME algorithm"""

import numpy as np
import sklearn

from newlime_types import IntArray
from newlime_utils import load_dataset, plot_weights
from sampler import Sampler


def main() -> None:
    """Main function"""

    # Load the dataset
    dataset = load_dataset("recidivism", "datasets", balance=True)

    # Get the target
    trg = dataset.test[0]

    # Create the sampler
    black_box = sklearn.ensemble.RandomForestClassifier()
    black_box.fit(dataset.train, dataset.labels_train)
    sampler = Sampler(trg, dataset.train, black_box, dataset.categorical_names)

    # Get the LIME explanation and plot the weights
    coef = explain(trg, sampler, 100000)
    plot_weights(coef, dataset.feature_names, img_name="weights.png")


def calc_weights(
    scaled_samples: IntArray, trg: IntArray, kernel_width: float | None = None
) -> list[float]:
    """
    Calculate the weights for the samples based on the distance to the target
    """

    # Calculate the distance between the samples and the target
    distances: list[float] = sklearn.metrics.pairwise_distances(
        scaled_samples, trg.reshape(1, -1)
    )[:, 0]

    def kernel(distance: float, kernel_width: float) -> float:
        return float(np.sqrt(np.exp(-(distance**2) / kernel_width**2)))

    # Calculate the weights
    if kernel_width is None:
        kernel_width = 0.75 * np.sqrt(scaled_samples.shape[1])
    weights = [kernel(x, kernel_width) for x in distances]

    return weights


def explain(
    trg: IntArray,
    sampler: Sampler,
    num_samples: int,
) -> list[float]:
    """Get the LIME explanation for the target"""

    # Sample and scale the data
    samples, labels = sampler.sample(num_samples)
    scaler = sklearn.preprocessing.StandardScaler().fit(samples)
    samples = scaler.transform(samples)

    # Calculate the weights
    weights = calc_weights(samples, trg)

    # Train a model on the samples
    surrogate = sklearn.linear_model.LogisticRegression()
    surrogate.fit(samples, labels, sample_weight=weights)

    # Standardize the coefficients
    coef: list[float] = surrogate.coef_[0]
    coef = coef / np.sum(np.abs(coef))

    return coef


if __name__ == "__main__":
    main()
