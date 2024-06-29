"""Test code for the implementation of R-LIME."""

from sklearn.ensemble import RandomForestClassifier

from src.rlime import rlime_lime, utils
from src.rlime.rlime import HyperParam, explain_instance
from src.rlime.rlime_types import Classifier, Dataset, IntArray
from src.rlime.sampler import Sampler
from src.rlime.utils import get_trg_sample


def main() -> None:
    """The main function of the module."""
    # Load the dataset.
    dataset = utils.load_dataset(
        "recidivism", "src/rlime/datasets/", balance=True
    )

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    idx = 0
    trg, _, _ = get_trg_sample(idx, dataset)  # type: ignore
    print(f"Target instance: {idx}")
    print(trg)

    # Test the LIME and R-LIME implementations.
    predict = lambda x: black_box.predict(x).astype(int)
    test_lime(trg, dataset, predict)
    for tau in [70, 80, 90]:
        test_rlime(trg, dataset, predict, HyperParam(tau=tau / 100))


def test_lime(trg: IntArray, dataset: Dataset, black_box: Classifier) -> None:
    """Generate the LIME explanation for the given sample."""
    print("LIME:")
    sampler = Sampler(trg, dataset.train, black_box, dataset.categorical_names)
    coef, _ = rlime_lime.explain(trg, sampler, 100000)
    print(f" Coefficients: {coef}")


def test_rlime(
    trg: IntArray,
    dataset: Dataset,
    black_box: Classifier,
    hyper_param: HyperParam,
) -> None:
    """Generate the R-LIME explanations for the given sample."""
    print(f"R-LIME (tau = {hyper_param.tau}):")
    result = explain_instance(trg, dataset, black_box, hyper_param)
    if result is None:
        print(" No explanation found.")
    else:
        names, arm = result
        weights = list(
            arm.surrogate_model["LogisticRegression"].weights.values()
        )
        weights = [w / sum(map(abs, weights)) for w in weights]
        print(f" Rule: {names}")
        print(f" Precision: {arm.n_rewards / arm.n_samples}")
        print(f" Coverage: {arm.coverage}")


if __name__ == "__main__":
    main()
