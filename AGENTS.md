# xgboost-predictor

Pure-JVM XGBoost predictor for online inference. No native library, no JNI. Reads XGBoost 3.x
UBJSON models and legacy binary dumps from XGBoost 1.0.0 through 2.x, and scores `FVec` feature
vectors.

Owned by Yelp.

## Layout

- `src/main/java/com/yelp/xgboost/` — prediction engine (vendored from
  [xgboost-predictor-java](https://github.com/komiya-atsushi/xgboost-predictor-java)).
  - `Predictor` — public API. `predict`/`predictSingle` (objective-transformed), `predictRaw`/
    `predictSingleRaw` (margin space), `predictLeaf`; each takes an optional `ntreeLimit`.
  - `FVec` — public feature-vector interface with static factories (`fromArray`, `fromMap`,
    `fromArrayWithZeroAsMissing`).
  - `gbm/`, `tree/`, `learner/` — booster, trees, objective transforms.
- `src/main/java/com/yelp/xgboost/parser/` — the XGBoost 3.x model reader.
  - `PredictorFactory` — entry point. Dispatches on the first byte (`{` = JSON/UBJSON).
  - `ModelReader` — internal legacy-binary stream reader (consumed by the boosters/trees).
  - `UBJSONReader` — hand-rolled UBJSON decoder for XGBoost's non-standard codec.
  - `XGBoostModelParser` — builds a `Predictor` from the parsed model tree.
  - `UValue` — format-agnostic value tree.
- `src/test/java/` + `src/test/resources/datasources/` — JUnit tests and golden fixtures.
  `PredictorBoosterParityTest` trains models with `xgboost4j` (test-only JNI dep) and checks the
  pure-JVM predictor matches the native Booster on the UBJSON path. This is where runtime parity is
  pinned. The legacy binary `ModelReader` path is pinned by frozen golden fixtures under
  `datasources/golden/v<version>/`, one folder per captured XGBoost version, with expected outputs
  and embedded feature vectors in that folder's `golden.json` (self-contained, no
  Spark/libsvm loading):
  - `GoldenValueParityTest` — the 2.0.3 baseline (`binf` magic, `major_version` 2), captured on
    `ml.dmlc:xgboost4j 2.0.3`.
  - `GoldenLegacy1xParityTest` — the two 1.x byte layouts Yelp ran in production: 1.0.0 (no magic)
    and 1.7.6 (`binf` magic, `major_version` 1), each including a `gblinear` case.
  1.x fixtures are regenerable with `datasources/golden/generate_legacy_goldens.py` under a matching
  `pip install xgboost==<version>` venv; 2.0.3 cannot be regenerated (native 3.3.0 cannot emit
  legacy binary), so those outputs are frozen literals.
- `src/jmh/java/` — the `me.champeau.jmh` benchmark comparing JNI vs pure-JVM single-row latency
  (`xgboost4j` is a benchmark-only dep, not on the runtime classpath).
- `src/demo/java/` + `src/demo/resources/` — `IrisDemo`, the `make demo` tutorial: trains an Iris
  model with `xgboost4j`, scores it with the pure-JVM predictor, and prints training metrics, a
  parity check, and single-call wall time. Its own source set (`xgboost4j` is a demo-only dep).

## Scope

`gbtree` only. `dart`, `gblinear`-from-JSON, and vector-leaf / multi-target trees are rejected at
load. `base_score` is stored in probability space and converted to margin space via the objective's
`probToMargin` before prediction.

## Build

JDK 17 required, Gradle 9 (wrapper checked in). `make build`, `make test`, `make bench`,
`make demo`, `make publish-local`. `make publish` publishes signed jars to Maven Central via the Central Portal
(`com.vanniktech.maven.publish`). It needs the Central Portal token and GPG key as Gradle
properties: `ORG_GRADLE_PROJECT_mavenCentralUsername`/`...Password` and
`ORG_GRADLE_PROJECT_signingInMemoryKey`/`...KeyPassword`. The legacy OSSRH host was shut down
2025-06-30.

## Conventions

- No inline comments. Put explanation at the doc level (class/method Javadoc) or use clear names.
- Keep the engine's fast paths intact: absent features must route the split default, never fall
  through to a numeric comparison. The `GoldenMissingParityTest` pins this against native xgboost
  ground truth (100/100 on both sparse and dense inputs).
