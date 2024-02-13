"""This module generates the experiment data for the paper."""

import csv
import multiprocessing

from sklearn.ensemble import RandomForestClassifier

import mylime
import newlime_base
import newlime_tabular
import newlime_utils
from newlime_tabular import Dataset
from newlime_types import Classifier, IntArray
from newlime_utils import RuleInfo, plot_weights
from sampler import Sampler


def sample_to_csv(
    tab: list[tuple[str, str]],
    path: str,
) -> None:
    """Save the sample as a CSV file."""
    with open(path, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for feature, sample in tab:
            writer.writerow([feature, sample])


def main() -> None:
    """The main function of the module."""

    # Load the dataset.
    dataset = newlime_utils.load_dataset(
        "recidivism", "datasets/", balance=True
    )

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    sample_num = 50
    idx_list = list(range(sample_num))
    trgs = []
    labels_trgs = []

    # Save the target instances as CSV files.
    for idx in idx_list:
        trg, label, tab = newlime_utils.get_trg_sample(idx, dataset)
        trgs.append(trg)
        labels_trgs.append(label)
        sample_to_csv(tab, f"output/{idx:04d}.csv")

    with multiprocessing.Pool() as pool:
        pool.starmap(
            generate_lime_and_rlime,
            [
                (idx, trg, dataset, black_box.predict)
                for idx, trg in zip(idx_list, trgs)
            ],
        )


def generate_lime_and_rlime(
    idx: int,
    trg: IntArray,
    dataset: newlime_tabular.Dataset,
    black_box: RandomForestClassifier,
) -> None:
    """Generate the LIME and R-LIME explanations for the given sample."""

    print(f"Target instance: {idx}")

    # Generate the LIME explanation and save it as an image.
    print("LIME")
    generate_lime(trg, dataset, black_box, f"output/lime-{idx:04d}.png")

    # Generate the R-LIME explanation and save it as an image.
    print("R-LIME")
    hyper_param = newlime_base.HyperParam()
    for hyper_param.tau in [0.7, 0.8, 0.9]:
        print(f"tau = {hyper_param.tau}")
        generate_rlime(
            trg,
            dataset,
            black_box,
            f"output/newlime-{idx:04d}-{int(hyper_param.tau * 100)}.png",
            hyper_param,
        )


def generate_lime(
    trg: IntArray,
    dataset: Dataset,
    black_box: Classifier,
    img_name: str,
) -> None:
    """Generate the LIME explanation for the given sample."""

    # Generate the LIME explanation.
    sampler = Sampler(trg, dataset.train, black_box, dataset.categorical_names)
    coef = mylime.explain(trg, sampler, 100000)

    # Save the LIME explanation as an image.
    plot_weights(
        coef, dataset.feature_names, rule_info=None, img_name=img_name
    )


def generate_rlime(
    trg: IntArray,
    dataset: Dataset,
    black_box: Classifier,
    img_name: str,
    hyper_param: newlime_base.HyperParam,
) -> None:
    """Generate the R-LIME explanations for the given sample."""

    # Generate the R-LIME explanation and standardize its weights.
    result = newlime_tabular.explain_instance(
        trg, dataset, black_box, hyper_param
    )
    if result is None:
        print("No explanation found.")
        return
    names, arm = result
    weights = list(arm.surrogate_model["LogisticRegression"].weights.values())
    weights = [w / sum(map(abs, weights)) for w in weights]

    # Save the R-LIME explanation as an image.
    rule_info = RuleInfo(names, arm.n_rewards / arm.n_samples, arm.coverage)
    plot_weights(weights, dataset.feature_names, rule_info, img_name)


if __name__ == "__main__":
    main()
