"""Test code for the implementation of R-LIME."""

from logging import getLogger

from sklearn.ensemble import RandomForestClassifier

from src.rlime import rlime_lime, utils
from src.rlime.rlime import HyperParam, explain_instance
from src.rlime.rlime_types import Classifier, Dataset, FloatArray, IntArray
from src.rlime.sampler import Sampler
from src.rlime.utils import get_trg_sample

# Set up the logger.
logger = getLogger()


def main() -> None:
    """The main function of the module."""
    # Load the dataset.
    dataset = utils.load_dataset("recidivism", balance=True)

    # Learn the black box model.
    black_box = RandomForestClassifier(n_estimators=100, n_jobs=1)
    black_box.fit(dataset.train, dataset.labels_train)

    # Get the target instances.
    idx = 0
    trg, _, _ = get_trg_sample(idx, dataset)
    logger.info("Target instance: %d", idx)
    logger.info(trg)

    # Test the LIME and R-LIME implementations.
    def predict(x: IntArray | FloatArray) -> IntArray:
        return black_box.predict(x).astype(int)

    test_lime(trg, dataset, predict)
    hyper_param = HyperParam()
    for hyper_param.tau in [0.70, 0.80, 0.90]:
        test_rlime(trg, dataset, predict, hyper_param)


def test_lime(trg: IntArray, dataset: Dataset, black_box: Classifier) -> None:
    """Generate the LIME explanation for the given sample."""
    logger.info("LIME:")
    sampler = Sampler(trg, dataset.train, black_box, dataset.categorical_names)
    coef, _ = rlime_lime.explain(trg, sampler, 100000)
    logger.info(" Coefficients: %f", coef)


def test_rlime(
    trg: IntArray,
    dataset: Dataset,
    black_box: Classifier,
    hyper_param: HyperParam,
) -> None:
    """Generate the R-LIME explanations for the given sample."""
    logger.info("R-LIME (tau = %f):", hyper_param.tau)
    result = explain_instance(trg, dataset, black_box, hyper_param)
    if result is None:
        logger.info(" No explanation found.")
    else:
        names, arm = result
        weights: list[float] = list(
            arm.surrogate_model["LogisticRegression"].weights.values()
        )
        weights = [w / sum(map(abs, weights)) for w in weights]
        logger.info(" Rule: %s", names)
        logger.info(" Precision: %f", arm.n_rewards / arm.n_samples)
        logger.info(" Coverage: %f", arm.coverage)


if __name__ == "__main__":
    main()
