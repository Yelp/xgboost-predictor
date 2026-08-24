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
package com.yelp.xgboost.demo;

import com.yelp.xgboost.FVec;
import com.yelp.xgboost.Predictor;
import com.yelp.xgboost.parser.PredictorFactory;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

/**
 * Runnable, hands-on tutorial: trains a real XGBoost model on the Iris dataset with the native
 * {@code xgboost4j} trainer, then scores it with this library's pure-JVM {@link Predictor}. It
 * prints the training curve, test accuracy, a worked prediction, a parity check against native, and
 * the wall time of a single prediction call each way.
 *
 * <p>This is a demo, not a benchmark. The single-call timings are illustrative wall clock, not JMH
 * measurements. Run it with {@code make demo}.
 */
public final class IrisDemo {

  private static final String[] SPECIES = {"setosa", "versicolor", "virginica"};
  private static final String[] FEATURE_NAMES = {
    "sepal_length", "sepal_width", "petal_length", "petal_width"
  };
  private static final int NUM_FEATURES = 4;
  private static final int NUM_CLASSES = 3;
  private static final int NUM_ROUNDS = 30;

  private IrisDemo() {}

  public static void main(String[] args) throws Exception {
    banner();

    List<float[]> features = new ArrayList<>();
    List<Integer> labels = new ArrayList<>();
    loadIris(features, labels);
    System.out.printf(
        "Loaded the Iris dataset: %d flowers, %d features, %d species.%n%n",
        features.size(), NUM_FEATURES, NUM_CLASSES);

    Split split = trainTestSplit(features, labels);
    System.out.printf(
        "Split into %d training rows and %d held-out test rows (stratified, seed-fixed).%n%n",
        split.trainRows.size(), split.testRows.size());

    section("1. Training an XGBoost model (native xgboost4j)");
    DMatrix trainMatrix = toDMatrix(split.trainRows, split.trainLabels);
    DMatrix testMatrix = toDMatrix(split.testRows, split.testLabels);
    Booster booster = train(trainMatrix, testMatrix, split);

    section("2. Test-set accuracy of the trained model");
    reportAccuracy(booster, split);

    section("3. Loading the model into the pure-JVM Predictor");
    Predictor predictor = loadIntoPredictor(booster);
    System.out.println(
        "Saved the booster to UBJSON bytes and loaded them with PredictorFactory.fromModelBytes.");
    System.out.println("No native library is needed from here on.\n");

    section("4. A worked prediction");
    workedPrediction(predictor, split);

    section("5. Parity check: pure-JVM Predictor vs native Booster");
    parityCheck(booster, predictor, split);

    section("6. Single-call wall time: native JNI vs pure-JVM");
    timing(booster, predictor, split);

    System.out.println(
        "That's it. Point PredictorFactory at your own model bytes and call predict(FVec).");
  }

  private static void banner() {
    System.out.println(
        "================================================================================");
    System.out.println(" xgboost-predictor: train with XGBoost, predict on the pure-JVM engine");
    System.out.println(
        "================================================================================\n");
  }

  private static void section(String title) {
    System.out.println(
        "--------------------------------------------------------------------------");
    System.out.println(title);
    System.out.println(
        "--------------------------------------------------------------------------");
  }

  /**
   * Trains one boosting round at a time so the demo can print a real training curve. The multiclass
   * log loss on the train and held-out test sets is computed here from each round's probabilities,
   * rather than relying on xgboost4j's built-in evaluation logging.
   */
  private static Booster train(DMatrix trainMatrix, DMatrix testMatrix, Split split)
      throws Exception {
    Map<String, Object> params = new HashMap<>();
    params.put("objective", "multi:softprob");
    params.put("num_class", NUM_CLASSES);
    params.put("eta", 0.3);
    params.put("max_depth", 3);
    params.put("tree_method", "hist");

    System.out.println(
        "Objective multi:softprob, " + NUM_ROUNDS + " boosting rounds, eta 0.3, max_depth 3.");
    System.out.println("Per-round multiclass log loss (lower is better):\n");
    System.out.printf("    %-7s %-14s %-14s%n", "round", "train mlogloss", "test mlogloss");

    Map<String, DMatrix> noWatches = new HashMap<>();
    Booster booster = null;
    for (int round = 1; round <= NUM_ROUNDS; round++) {
      booster = XGBoost.train(trainMatrix, params, 1, noWatches, null, null, null, 0, booster);
      if (round == 1 || round % 5 == 0 || round == NUM_ROUNDS) {
        double trainLoss = mlogloss(booster, trainMatrix, split.trainLabels);
        double testLoss = mlogloss(booster, testMatrix, split.testLabels);
        System.out.printf("    %-7d %-14.4f %-14.4f%n", round, trainLoss, testLoss);
      }
    }
    System.out.println();
    return booster;
  }

  private static double mlogloss(Booster booster, DMatrix matrix, List<Integer> labels)
      throws Exception {
    float[][] probs = booster.predict(matrix);
    double sum = 0.0;
    for (int i = 0; i < probs.length; i++) {
      double p = Math.max(probs[i][labels.get(i)], 1e-15);
      sum += -Math.log(p);
    }
    return sum / probs.length;
  }

  private static void reportAccuracy(Booster booster, Split split) throws Exception {
    DMatrix testMatrix = toDMatrix(split.testRows, split.testLabels);
    float[][] probs = booster.predict(testMatrix);
    int correct = 0;
    for (int i = 0; i < probs.length; i++) {
      if (argmax(probs[i]) == split.testLabels.get(i)) {
        correct++;
      }
    }
    System.out.printf(
        "Correctly classified %d of %d held-out flowers (%.1f%% accuracy).%n%n",
        correct, probs.length, 100.0 * correct / probs.length);
  }

  private static Predictor loadIntoPredictor(Booster booster) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    booster.saveModel(out, "ubj");
    return PredictorFactory.fromModelBytes(out.toByteArray());
  }

  private static void workedPrediction(Predictor predictor, Split split) {
    int idx = 0;
    float[] row = split.testRows.get(idx);
    int actual = split.testLabels.get(idx);

    FVec features = FVec.fromArray(row);
    float[] probabilities = predictor.predict(features);
    int predicted = argmax(probabilities);

    System.out.println("Take one held-out flower and score it with the pure-JVM Predictor:\n");
    for (int f = 0; f < NUM_FEATURES; f++) {
      System.out.printf("    %-13s = %.1f cm%n", FEATURE_NAMES[f], row[f]);
    }
    System.out.println("\n  Predicted class probabilities:");
    for (int c = 0; c < NUM_CLASSES; c++) {
      System.out.printf(
          "    %-11s %6.2f%%  %s%n", SPECIES[c], 100.0 * probabilities[c], bar(probabilities[c]));
    }
    System.out.printf(
        "%n  Predicted: %s   Actual: %s   %s%n%n",
        SPECIES[predicted], SPECIES[actual], predicted == actual ? "(correct)" : "(miss)");
  }

  private static void parityCheck(Booster booster, Predictor predictor, Split split)
      throws Exception {
    double maxDiff = 0.0;
    for (int i = 0; i < split.testRows.size(); i++) {
      float[] row = split.testRows.get(i);
      float[] jvm = predictor.predict(FVec.fromArray(row));
      float[] jni = booster.predict(new DMatrix(row, 1, NUM_FEATURES, Float.NaN), false, 0)[0];
      for (int c = 0; c < NUM_CLASSES; c++) {
        maxDiff = Math.max(maxDiff, Math.abs(jvm[c] - jni[c]));
      }
    }
    System.out.printf(
        "Scored all %d test rows both ways. Largest probability difference: %.2e.%n",
        split.testRows.size(), maxDiff);
    System.out.println("The pure-JVM engine reproduces native XGBoost to floating-point noise.\n");
  }

  private static void timing(Booster booster, Predictor predictor, Split split) throws Exception {
    float[] row = split.testRows.get(0);
    FVec features = FVec.fromArray(row);

    warmUp(booster, predictor, row, features);

    long jniStart = System.nanoTime();
    booster.predict(new DMatrix(row, 1, NUM_FEATURES, Float.NaN), false, 0);
    long jniNanos = System.nanoTime() - jniStart;

    long jvmStart = System.nanoTime();
    predictor.predict(features);
    long jvmNanos = System.nanoTime() - jvmStart;

    System.out.println("Wall time of one single-row prediction call (after warm-up):\n");
    System.out.printf("    native JNI Booster   %,10.1f us%n", jniNanos / 1000.0);
    System.out.printf("    pure-JVM Predictor   %,10.1f us%n", jvmNanos / 1000.0);
    if (jvmNanos > 0) {
      System.out.printf(
          "%n  Pure-JVM was about %.0fx faster on this call.%n", (double) jniNanos / jvmNanos);
    }
    System.out.println(
        "  The JNI path pays per-call DMatrix allocation and a native round-trip; the pure-JVM");
    System.out.println(
        "  path scores the FVec directly. For per-row online inference this is the whole point.\n");
  }

  private static void warmUp(Booster booster, Predictor predictor, float[] row, FVec features)
      throws Exception {
    for (int i = 0; i < 200; i++) {
      predictor.predict(features);
      booster.predict(new DMatrix(row, 1, NUM_FEATURES, Float.NaN), false, 0);
    }
  }

  private static DMatrix toDMatrix(List<float[]> rows, List<Integer> labels) throws Exception {
    List<LabeledPoint> points = new ArrayList<>(rows.size());
    for (int i = 0; i < rows.size(); i++) {
      points.add(new LabeledPoint(labels.get(i), NUM_FEATURES, null, rows.get(i)));
    }
    return new DMatrix(points.iterator(), null);
  }

  private static void loadIris(List<float[]> features, List<Integer> labels) throws Exception {
    try (InputStream in = IrisDemo.class.getClassLoader().getResourceAsStream("iris.csv")) {
      if (in == null) {
        throw new IllegalStateException("iris.csv resource not found on the classpath");
      }
      BufferedReader reader = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
      reader.readLine(); // header
      String line;
      while ((line = reader.readLine()) != null) {
        if (line.isBlank()) {
          continue;
        }
        String[] parts = line.split(",");
        float[] row = new float[NUM_FEATURES];
        for (int f = 0; f < NUM_FEATURES; f++) {
          row[f] = Float.parseFloat(parts[f]);
        }
        features.add(row);
        labels.add(speciesIndex(parts[NUM_FEATURES]));
      }
    }
  }

  /**
   * A deterministic stratified split: every fifth row of each species goes to the test set. Iris is
   * ordered by species (50 each), so this keeps all three classes represented on both sides.
   */
  private static Split trainTestSplit(List<float[]> features, List<Integer> labels) {
    Split split = new Split();
    for (int i = 0; i < features.size(); i++) {
      if (i % 5 == 0) {
        split.testRows.add(features.get(i));
        split.testLabels.add(labels.get(i));
      } else {
        split.trainRows.add(features.get(i));
        split.trainLabels.add(labels.get(i));
      }
    }
    return split;
  }

  private static int speciesIndex(String name) {
    for (int i = 0; i < SPECIES.length; i++) {
      if (SPECIES[i].equals(name)) {
        return i;
      }
    }
    throw new IllegalArgumentException("Unknown species: " + name);
  }

  private static int argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  private static String bar(float fraction) {
    int filled = Math.round(fraction * 30);
    return "#".repeat(filled);
  }

  private static final class Split {
    final List<float[]> trainRows = new ArrayList<>();
    final List<Integer> trainLabels = new ArrayList<>();
    final List<float[]> testRows = new ArrayList<>();
    final List<Integer> testLabels = new ArrayList<>();
  }
}
