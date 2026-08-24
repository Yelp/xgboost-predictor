/*
 * Copyright 2026 Yelp Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.yelp.xgboost.benchmark;

import com.yelp.xgboost.FVec;
import com.yelp.xgboost.Predictor;
import com.yelp.xgboost.parser.PredictorFactory;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Compares single-row prediction latency of the JNI xgboost4j Booster against the pure-JVM
 * predictor on the SAME model and input. This is a non-blocking, on-demand tool (not a test, not
 * CI-gated). Run with:
 *
 * <pre>./gradlew jmh</pre>
 *
 * The headline is per-row prediction, where the JNI + per-call DMatrix overhead dominates.
 */
@BenchmarkMode({Mode.AverageTime, Mode.Throughput})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(2)
public class XGBoostPredictionBenchmark {

  private Booster booster;
  private Predictor predictor;
  private float[] featureValues;
  private FVec featureVec;
  private int numFeatures;

  @Setup
  public void setup() throws Exception {
    numFeatures = 30;
    int rows = 512;
    Random random = new Random(42L);

    List<LabeledPoint> labeledPoints = new ArrayList<>(rows);
    for (int i = 0; i < rows; i++) {
      float[] values = new float[numFeatures];
      float sum = 0f;
      for (int j = 0; j < numFeatures; j++) {
        values[j] = random.nextFloat();
        sum += values[j];
      }
      float label = sum > numFeatures / 2.0f ? 1.0f : 0.0f;
      labeledPoints.add(new LabeledPoint(label, numFeatures, null, values));
    }
    DMatrix trainMatrix = new DMatrix(labeledPoints.iterator(), null);

    Map<String, Object> params = new HashMap<>();
    params.put("objective", "binary:logistic");
    params.put("eta", 0.3);
    params.put("max_depth", 6);
    params.put("tree_method", "hist");
    booster = XGBoost.train(trainMatrix, params, 50, new HashMap<>(), null, null);

    ByteArrayOutputStream out = new ByteArrayOutputStream();
    booster.saveModel(out);
    predictor = PredictorFactory.fromModelBytes(out.toByteArray());

    featureValues = new float[numFeatures];
    for (int j = 0; j < numFeatures; j++) {
      featureValues[j] = random.nextFloat();
    }
    featureVec = FVec.fromArray(featureValues);
  }

  @Benchmark
  public void jniBoosterSingleRow(Blackhole bh) throws Exception {
    DMatrix dmatrix = new DMatrix(featureValues, 1, numFeatures, Float.NaN);
    float[][] result = booster.predict(dmatrix, false, 0);
    bh.consume(result[0][0]);
  }

  @Benchmark
  public void pureJvmPredictorSingleRow(Blackhole bh) {
    float[] result = predictor.predict(featureVec);
    bh.consume(result[0]);
  }
}
