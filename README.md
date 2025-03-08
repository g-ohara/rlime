# R-LIME

This is an implementation of **R-LIME**, a novel method for interpreting ML models, proposed in our paper[^1].
R-LIME explains behavior of black-box classifiers such as deep neural networks or ensemble models.
It linearly approximizes a decision boundary of the black-box classifier in a local rectangular region, and maximizes the region as long as the approximation accuracy is higher than a given threshold.
Then, it provides contribution of each feature to the prediction and rule that restricted the approximation region.

[^1]: Ohara, G., Kimura, K., Kudo, M. (2025). R-LIME: Rectangular Constraints and Optimization for Local Interpretable Model-agnostic Explanation Methods. In: Antonacopoulos, A., Chaudhuri, S., Chellappa, R., Liu, CL., Bhattacharya, S., Pal, U. (eds) Pattern Recognition. ICPR 2024. Lecture Notes in Computer Science, vol 15315. Springer, Cham. https://doi.org/10.1007/978-3-031-78354-8_6

### Related repositories

- [g-ohara/rlime-examples](https://github.com/g-ohara/rlime-examples)
- [g-ohara/rlime-paper](https://github.com/g-ohara/rlime-paper)
- [g-ohara/rlime-poster](https://github.com/g-ohara/rlime-poster)
- [g-ohara/rlime-slides](https://github.com/g-ohara/rlime-slides)
- [g-ohara/rlime-ga](https://github.com/g-ohara/rlime-ga)

## Installation
Install the package:
```sh
pip install rlime
```
## Usage
Example code:
```py
from rlime.rlime import HyperParam, explain_instance
from rlime.utils import load_dataset, get_trg_sample
from sklearn.ensemble import RandomForestClassifier

# Load dataset and sample a focal point
dataset = load_dataset("recidivism", balance=True)
focal_point, _, _ = get_trg_sample(0, dataset)

# Train black-box model (random forest)
black_box = RandomForestClassifier(n_jobs=-1)
black_box.fit(dataset.train, dataset.labels_train)

# Generate explanation
hyper_param = HyperParam()
hyper_param.tau=0.70
rlime_exp = explain_instance(focal_point, dataset, black_box.predict, hyper_param)

# Print generated explanation if found
if rlime_exp is None:
  print("No explanation found")
else:
  rule, arm = rlime_exp
  print(f"Rule: {rule}")
  print("Weights:")
  weights = arm.surrogate_model["LogisticRegression"].weights.values()
  sum_weights = sum([abs(w) for w in weights])
  weights = [w / sum_weights for w in weights]
  top_5 = sorted(enumerate(weights), key=lambda x: abs(x[1]), reverse=True)[:5]
  for i, w in top_5:
    print(f"  {dataset.feature_names[i]:^16}: {w:+.4f}")
  print(f"Accuracy: {arm.n_rewards / arm.n_samples:.4f}")
  print(f"Coverage: {arm.coverage:.4f}")
```
You will get output:
```
Rule: ['Priors = 1']
Weights:
  PrisonViolations: +0.1789
        Age       : -0.1526
    MonthsServed  : +0.1267
        Race      : -0.1191
      Married     : -0.1010
Accuracy: 0.7054
Coverage: 0.4396
```

## Licence
[MIT](https://github.com/g-ohara/rlime/blob/main/LICENSE)

## Author
[Genji Ohara](https://github.com/g-ohara)
