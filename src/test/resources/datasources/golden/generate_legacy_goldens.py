#!/usr/bin/env python3
"""Generate frozen legacy-binary golden fixtures for a given XGBoost 1.x version.

Yelp production models span xgboost 1.0.0 through 1.7.6. That range emits two
distinct legacy-binary layouts, both parsed by the pure-JVM ModelReader:

  * 1.0.0            no "binf" magic, major_version == 1  (ModelParam else-branch)
  * 1.3.0 .. 1.7.6   "binf" magic present, major_version == 1  (ModelParam if-branch)

Native xgboost 3.x can neither read nor emit either layout, so these fixtures are
captured once per version and frozen. Unlike the 2.0.3 baseline, 1.x wheels are
still installable from PyPI, so this generator is committed for reproducibility.

Usage (run under a venv with the target xgboost pinned):
    pip install "xgboost==1.7.6" "numpy<2"
    python generate_legacy_goldens.py 1.7.6

It writes:
    golden/v<version>/<objective>.model      (legacy binary model bytes)
    golden/v<version>/golden.json            (embedded datasets + frozen outputs)

The paired GoldenValueParityTest loads the JSON, re-runs the pure-JVM predictor,
and asserts it reproduces these frozen native outputs. Running that test is the
parity gate: a mismatch means the predictor diverged from native xgboost.
"""

import json
import os
import sys

import numpy as np
import xgboost as xgb

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_ROWS = 5


def binomial_dataset():
    rng = np.random.RandomState(42)
    x = rng.rand(80, 8).astype(np.float32)
    y = (x.sum(axis=1) > 4.0).astype(np.float32)
    return x, y


def multinomial_dataset():
    rng = np.random.RandomState(7)
    x = rng.rand(90, 4).astype(np.float32)
    y = (x[:, 0] * 3).astype(int).clip(0, 2).astype(np.float32)
    return x, y


def positive_dataset():
    rng = np.random.RandomState(19)
    x = rng.rand(80, 8).astype(np.float32)
    y = (1.0 + x[:, 0] * 3.0 + x[:, 1] * 2.0).astype(np.float32)
    return x, y


def ranking_dataset():
    rng = np.random.RandomState(11)
    x = rng.rand(80, 6).astype(np.float32)
    y = np.floor(x[:, 0] * 3 + x[:, 1]).astype(int).clip(0, 3).astype(np.float32)
    group = [20, 20, 20, 20]
    return x, y, group


def train(objective, booster, x, y, num_class, group=None):
    params = {
        "objective": objective,
        "booster": booster,
        "seed": 0,
        "base_score": 0.5,
        "max_depth": 3,
    }
    if objective.startswith("multi:"):
        params["num_class"] = num_class
    if objective.startswith("rank:"):
        params["min_child_weight"] = 0.1
        params["reg_lambda"] = 0.0
    dm = xgb.DMatrix(x, label=y)
    if group is not None:
        dm.set_group(group)
    return xgb.train(params, dm, num_boost_round=5)


def argmax(values):
    return int(np.argmax(values))


def dense_rows(x):
    return {str(i): [float(v) for v in x[i]] for i in range(NUM_ROWS)}


def sparse_rows(x):
    rows = {}
    for i in range(NUM_ROWS):
        rows[str(i)] = {str(j): float(x[i][j]) for j in range(x.shape[1])}
    return rows


def case_outputs(bst, x, booster, objective, num_class):
    dm = xgb.DMatrix(x[:NUM_ROWS])
    raw = bst.predict(dm)
    margin = bst.predict(dm, output_margin=True)
    leaf = None if booster == "gblinear" else bst.predict(dm, pred_leaf=True)

    predictions, margins, probabilities, leaf_indices = [], [], [], []
    for i in range(NUM_ROWS):
        if num_class > 2 and objective == "multi:softprob":
            predictions.append(argmax(raw[i]))
            probabilities.append([float(v) for v in raw[i]])
            margins.append([float(v) for v in margin[i]])
        elif objective == "multi:softmax":
            predictions.append(float(raw[i]))
            margins.append([float(v) for v in margin[i]])
        elif num_class == 2 or objective in ("binary:logistic",):
            p = float(raw[i])
            predictions.append(p)
            probabilities.append([1.0 - p, p])
            margins.append([float(margin[i])])
        else:
            predictions.append(float(raw[i]))
            margins.append([float(margin[i])])
        if leaf is not None:
            leaf_indices.append([int(v) for v in np.atleast_1d(leaf[i])])
    return predictions, margins, probabilities, leaf_indices


def main():
    version = sys.argv[1]
    assert xgb.__version__ == version, f"env has {xgb.__version__}, expected {version}"

    x_bin, y_bin = binomial_dataset()
    x_multi, y_multi = multinomial_dataset()
    x_pos, y_pos = positive_dataset()
    x_rank, y_rank, group_rank = ranking_dataset()

    model_dir = os.path.join(HERE, f"v{version}")
    os.makedirs(model_dir, exist_ok=True)

    specs = [
        ("reg_squarederror", "reg:squarederror", "gbtree", "binomial", 0),
        ("binary_logistic", "binary:logistic", "gbtree", "binomial", 2),
        ("reg_logistic", "reg:logistic", "gbtree", "binomial", 2),
        ("multi_softprob", "multi:softprob", "gbtree", "multinomial", 3),
        ("multi_softmax", "multi:softmax", "gbtree", "multinomial", 3),
        ("reg_tweedie", "reg:tweedie", "gbtree", "positive", 0),
        ("rank_ndcg", "rank:ndcg", "gbtree", "ranking", 0),
        ("gblinear_logistic", "binary:logistic", "gblinear", "binomial", 2),
        ("gblinear_softprob", "multi:softprob", "gblinear", "multinomial", 3),
    ]

    data = {
        "binomial": (x_bin, y_bin, None),
        "multinomial": (x_multi, y_multi, None),
        "positive": (x_pos, y_pos, None),
        "ranking": (x_rank, y_rank, group_rank),
    }

    cases = []
    for name, objective, booster, dataset, num_class in specs:
        x, y, group = data[dataset]
        bst = train(objective, booster, x, y, num_class, group)
        rel = f"datasources/golden/v{version}/{name}.model"
        bst.save_model(os.path.join(HERE, f"v{version}/{name}.model"))
        preds, margins, probs, leaves = case_outputs(bst, x, booster, objective, num_class)
        cases.append(
            {
                "modelResource": rel,
                "objective": objective,
                "booster": booster,
                "numClasses": num_class,
                "numFeatures": int(x.shape[1]),
                "dataset": dataset,
                "rowIndices": list(range(NUM_ROWS)),
                "predictions": preds,
                "margins": margins,
                "probabilities": probs,
                "leafIndices": leaves,
            }
        )

    magic = "absent ('binf' added in 1.3.0)" if version == "1.0.0" else "present ('binf')"
    out = {
        "xgboostVersion": version,
        "producer": f"xgboost=={version} legacy binary format, magic {magic}, major_version 1",
        "note": (
            "Frozen legacy-binary golden captured on xgboost "
            f"{version}. Regenerate with generate_legacy_goldens.py under a matching "
            "venv. Feature vectors are embedded so the test is self-contained."
        ),
        "datasets": {
            "binomial": dense_rows(x_bin),
            "multinomial": sparse_rows(x_multi),
            "positive": dense_rows(x_pos),
            "ranking": dense_rows(x_rank),
        },
        "cases": cases,
    }
    out_path = os.path.join(HERE, f"v{version}", "golden.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path} and {len(cases)} models under v{version}/")


if __name__ == "__main__":
    main()
