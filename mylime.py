"""This module contains the LIME algorithm"""

import lime
import numpy as np
import sklearn

from newlime_types import IntArray
from newlime_utils import load_dataset
from sampler import Sampler


def main() -> None:
    """Main function. Test whether weights calculated by this module equal to
    weights calculated by the original LIME implementation."""

    # Load the dataset
    dataset = load_dataset("recidivism", "datasets", balance=True)

    for trg in dataset.test:

        # Create the sampler
        black_box = sklearn.ensemble.RandomForestClassifier()
        black_box.fit(dataset.train, dataset.labels_train)
        sampler = Sampler(
            trg, dataset.train, black_box.predict, dataset.categorical_names
        )

        num_samples = 10000

        # Get the LIME explanation twice by this module
        coef1 = np.array(explain(trg, sampler, num_samples))
        coef2 = np.array(explain(trg, sampler, num_samples))
        diff_twice = float(np.linalg.norm(coef1 - coef2))

        # Get the LIME explanation by the original LIME implementation
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            dataset.train,
            discretize_continuous=False,
        )
        lime_exp = lime_explainer.explain_instance(
            trg,
            black_box.predict_proba,
            num_features=15,
            top_labels=1,
            num_samples=num_samples,
        )
        pred_label = black_box.predict(trg.reshape(1, -1))[0]
        coef_org = [0.0] * len(dataset.feature_names)
        for t in lime_exp.local_exp[pred_label]:
            coef_org[t[0]] = t[1] * (pred_label * 2 - 1)
        coef_org = coef_org / np.sum(np.abs(coef_org))
        diff_org = float(np.linalg.norm(coef1 - coef_org))

        # Compare the weights
        print(f"{diff_twice:.4f} {diff_org:.4f}")


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
