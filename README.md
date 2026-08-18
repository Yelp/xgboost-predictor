<p align="center">
  <img src="docs/banner.jpg" alt="xgboost-predictor" width="100%"/>
</p>

# xgboost-predictor

[![Test](https://github.com/Yelp/xgboost-predictor/actions/workflows/test.yml/badge.svg)](https://github.com/Yelp/xgboost-predictor/actions/workflows/test.yml)

Pure-JVM XGBoost predictor for online inference. Reads XGBoost 3.x UBJSON models and legacy binary
dumps from XGBoost 1.x through 2.x, and scores feature vectors with no native library, no JNI, and
no runtime dependencies.

## Supported model versions

The current UBJSON format (XGBoost 3.x, written by `Booster.saveModel`) and the legacy binary format
back to **XGBoost 1.0.0** load through the same engine. The legacy path is pinned by frozen golden
fixtures for both byte layouts it covers: 1.0.0 (no `binf` magic) and 1.3.0 through 1.7.6 (`binf`
magic), plus a 2.0.3 baseline. XGBoost 0.x binary dumps are not supported.

## Why this exists

The `xgboost4j` Booster predicts through JNI. Every call marshals features into a fresh `DMatrix`
and crosses the JNI boundary, so per-row overhead dominates online serving. It also ships a
platform-specific `.so`/`.dylib` that complicates packaging for JVM services.

This predictor keeps only the scoring hot path in pure Java: tree traversal over `int[]`-packed
nodes and an `FVec` feature view. No native library, no JNI, no per-call `DMatrix`. Training and
batch scoring stay with `xgboost4j`. This is for latency-sensitive per-row inference.

## Benchmark

`XGBoostPredictionBenchmark` (JMH) compares single-row latency against the JNI Booster on the same
model (30 features, `binary:logistic`, depth 6, 50 trees) and input. Representative results
(JDK 17, `AverageTime` mode, lower is better):

| Benchmark                     | Score        |
| ----------------------------- | ------------ |
| `jniBoosterSingleRow`         | ~2000+ µs/op |
| `pureJvmPredictorSingleRow`   | ~0.58 µs/op  |

The pure-JVM path is orders of magnitude faster per row because it avoids the JNI crossing and the
per-call `DMatrix` allocation. Absolute numbers vary by machine, the gap is the point. Run it with
`make bench`. `xgboost4j` is a benchmark-only dependency, not on the runtime classpath.

## Try it

`make demo` runs an end-to-end tutorial: it trains a model on the Iris dataset, scores it with this
predictor, checks parity against native, and times a single call each way. Read the console output
and start from there. The code is in `src/demo/java`.

## Coordinates

```
com.yelp:xgboost-predictor:1.0.0
```

## Usage

```java
import com.yelp.xgboost.Predictor;
import com.yelp.xgboost.parser.PredictorFactory;
import com.yelp.xgboost.FVec;

import java.io.InputStream;
import java.util.Map;

try (InputStream model = Files.newInputStream(Path.of("model.ubj"))) {
    Predictor predictor = PredictorFactory.fromModelStream(model);

    FVec features = FVec.fromMap(Map.of(0, 0.3f, 2, 13.0f));
    float[] prediction = predictor.predict(features);
}
```

`PredictorFactory` dispatches on the model's first byte, not the filename, so the extension does not
matter. A leading `{` marks a UBJSON document, anything else is treated as the legacy binary format.
UBJSON models are scoped to `gbtree`. `dart`, `gblinear`, and vector-leaf or multi-target trees are
rejected at load. Legacy 1.x binary bundles carrying a `gblinear` booster still load, since they
predate that scope restriction.

### Missing values

XGBoost routes a missing feature to a split's default child. `FVec.fromMap` treats any absent index
as missing. For dense inputs, `fromArray` treats only NaN as missing, and
`fromArrayWithZeroAsMissing` also treats zeros as missing (for a model trained with `missing=0`).

For raw margin-space scores instead of the objective-transformed prediction, use `predictRaw` (or
`predictSingleRaw` for single-value models) in place of `predict`.

## Build

```
make build          # compile
make test           # run the JUnit suite
make demo           # run the end-to-end Iris tutorial
make publish-local  # install to ~/.m2
```

Requires JDK 17. The Gradle wrapper (`./gradlew`) is checked in.

## Publishing

`make publish` publishes signed jars to Maven Central through the Central Portal. It needs the
Central Portal token and GPG key as Gradle properties: `mavenCentralUsername`, `mavenCentralPassword`,
`signingInMemoryKey`, and `signingInMemoryKeyPassword` (each prefixed with `ORG_GRADLE_PROJECT_` as an
environment variable). The GitHub Actions `release` workflow runs it automatically when a GitHub
Release is published.

## License

Apache License 2.0. The predictor engine is vendored from
[xgboost-predictor-java](https://github.com/komiya-atsushi/xgboost-predictor-java). See `NOTICE`.
