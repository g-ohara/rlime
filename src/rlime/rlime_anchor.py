"""This module contains the function for running Anchor[1].

[1] Ribeiro, M. T., Singh, S., & Guestrin, C. (2018).
    Anchors: High-Precision Model-Agnostic Explanations.
    AAAI Conference on Artificial Intelligence (AAAI).
"""

from typing import TYPE_CHECKING

from anchor.anchor_tabular import AnchorTabularExplainer

from .rlime_types import Classifier, Dataset, IntArray

if TYPE_CHECKING:
    from anchor.anchor_explanation import AnchorExplanation


def anchor(
    trg: IntArray, dataset: Dataset, classifier: Classifier, threshold: float
) -> tuple[list[str], float, float]:
    """Run Anchor and print the rule, precision and coverage on the terminal.

    Parameters
    ----------
    trg : np.ndarray
        The target sample
    dataset : Dataset
        The dataset
    classifier : Classifier
        The black box classifier
    threshold : float
        The threshold

    Returns:
    -------
    tuple[str, float, float]
        The threshold, rule, precision and coverage
    """
    anchor_explainer = AnchorTabularExplainer(
        dataset.class_names,
        dataset.feature_names,
        dataset.train,
        dataset.categorical_names,
    )

    anchor_exp: AnchorExplanation
    anchor_exp = anchor_explainer.explain_instance(trg, classifier, threshold)

    rule: list[str] = anchor_exp.names()
    if not isinstance(rule, list):
        msg = "Rule should be list type."
        raise TypeError(msg)

    acc: float = anchor_exp.precision()
    cov: float = anchor_exp.coverage()
    if not isinstance(acc, float) or not isinstance(cov, float):
        msg = "Accuracy and Coverage should be float type."
        raise TypeError(msg)

    return rule, acc, cov
