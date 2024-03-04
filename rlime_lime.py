"""This module contains the LIME algorithm"""

import lime
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm  # type: ignore

from rlime_types import IntArray
from sampler import Sampler
from utils import load_dataset


def main() -> None:
    """Main function. Test whether weights calculated by this module equal to
    weights calculated by the original LIME implementation."""

    def original_lime(
        trg: IntArray,
        train_data: IntArray,
        black_box: sklearn.ensemble.RandomForestClassifier,
        num_samples: int,
    ) -> list[float]:

        # Get the LIME explanation by the original LIME implementation
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            train_data,
            discretize_continuous=False,
        )
        lime_exp = lime_explainer.explain_instance(
            trg,
            black_box.predict_proba,
            num_features=15,
            num_samples=num_samples,
        )
        coef_org = [0.0] * len(dataset.feature_names)
        for t in lime_exp.local_exp[1]:
            coef_org[t[0]] = t[1]
        coef_org = coef_org / np.sum(np.abs(coef_org))

        return coef_org

    # Load the dataset
    dataset = load_dataset("recidivism", "datasets", balance=True)

    num_samples = 10000

    diffs = []
    my_accs = []
    org_accs = []
    for trg in tqdm(dataset.test[:100]):

        # Create the sampler
        black_box = sklearn.ensemble.RandomForestClassifier()
        black_box.fit(dataset.train, dataset.labels_train)
        sampler = Sampler(
            trg, dataset.train, black_box.predict, dataset.categorical_names
        )

        # Get the LIME explanations by the two methods
        my_coef, scaler = explain(trg, sampler, num_samples)
        org_coef = original_lime(trg, dataset.train, black_box, num_samples)

        # Compare the accuracy of the two methods
        samples, labels = sampler.sample(num_samples)
        samples = scaler.transform(samples)
        my_acc = np.mean((my_coef @ samples.T >= 0) == labels)
        org_acc = np.mean((org_coef @ samples.T >= 0) == labels)
        my_accs.append(my_acc)
        org_accs.append(org_acc)
        diff = my_acc - org_acc
        diffs.append(diff)

    print("Differences in accuracy:")
    print(f"Ave: {np.mean(diffs):+.6f}")
    print(f"Max: {np.max(diffs):+.6f}")
    print(f"Min: {np.min(diffs):+.6f}")
    print(f"Std: {np.std(diffs):+.6f}")

    print("Correlation between the two methods:")
    print(f"{pd.Series(my_accs).corr(pd.Series(org_accs)):.4f}")


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
) -> tuple[list[float], sklearn.preprocessing.StandardScaler]:
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

    return coef, scaler


if __name__ == "__main__":
    main()
