"""Module for plotting images."""

import csv

from newlime_utils import RuleInfo, load_dataset, plot_weights


def main() -> None:
    """The main function of the module."""

    # Load the dataset.
    dataset = load_dataset("recidivism", "datasets/", balance=True)

    for idx in range(50):

        # Load the weights.
        csv_name = f"output/lime-{idx:04d}.csv"
        result = load_weights(csv_name)
        if result is not None:
            weights, _ = result

            # Plot the weights.
            img_name = f"output/lime-{idx:04d}.eps"
            plot_weights(weights, dataset.feature_names, img_name=img_name)

        for tau in [70, 80, 90]:

            # Load the weights.
            csv_name = f"output/newlime-{idx:04d}-{tau}.csv"
            result = load_weights(csv_name)
            if result is not None:
                weights, rule_info = result

                # Plot the weights.
                img_name = f"output/newlime-{idx:04d}-{tau}.eps"
                plot_weights(
                    weights,
                    dataset.feature_names,
                    rule_info,
                    img_name=img_name,
                )


def load_weights(
    path: str,
) -> tuple[list[float], RuleInfo | None] | None:
    """Load the weights from a CSV file.

    Parameters
    ----------
    path : str
        The path to the CSV file.

    Returns
    -------
    tuple[list[float], RuleInfo | None]
        The weights and the rule information.
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            weights = list(map(float, next(reader)))
            rule_info = None
            try:
                rule_str = next(reader)
                coverage, precision = next(reader)
                rule_info = RuleInfo(
                    rule_str, float(precision), float(coverage)
                )
            except StopIteration:
                pass
            return weights, rule_info
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()
