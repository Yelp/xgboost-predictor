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
package com.yelp.xgboost;

import static com.yelp.xgboost.testing.BoosterTestUtils.CATEGORICAL_NUM_FEATURES;
import static com.yelp.xgboost.testing.BoosterTestUtils.NUM_FEATURES;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import com.yelp.xgboost.testing.BoosterTestUtils;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import org.junit.Test;

/**
 * Predictor-vs-Booster parity: trains a model with xgboost4j, loads it through {@link
 * com.yelp.xgboost.parser.PredictorFactory}, and checks the pure-JVM {@link Predictor} reproduces
 * the native Booster's margin, probability, and leaf indices on identical inputs. Covers binary,
 * multinomial, reg:gamma, categorical splits (low- and high-cardinality), deep many-tree models,
 * and missing-value routing (dense NaN and omitted sparse indices). Parity for the runtime lives
 * here, not in downstream consumers.
 */
public class PredictorBoosterParityTest {

  private static final double EPS = 1e-5;

  @Test
  public void leafIndicesMatchForBinaryModel() throws Exception {
    Random random = new Random(11L);
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster booster = BoosterTestUtils.trainBinaryBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int i = 0; i < 30; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      int[] boosterLeaves = toInts(nativeLeaf(booster, row, NUM_FEATURES));
      int[] predictorLeaves = predictor.predictLeaf(FVec.fromArray(row));
      assertArrayEquals("leaf mismatch on row " + i, boosterLeaves, predictorLeaves);
    }
  }

  @Test
  public void leafIndicesMatchForMultinomialModel() throws Exception {
    Random random = new Random(12L);
    DMatrix dataset = BoosterTestUtils.multinomialDataset();
    Booster booster = BoosterTestUtils.trainMultinomialBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int i = 0; i < 30; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      int[] boosterLeaves = toInts(nativeLeaf(booster, row, NUM_FEATURES));
      int[] predictorLeaves = predictor.predictLeaf(FVec.fromArray(row));
      assertArrayEquals("leaf mismatch on row " + i, boosterLeaves, predictorLeaves);
    }
  }

  /**
   * Per-class margin and probability parity for multiclass. XGBoost 3.x stores base_score as a
   * per-class vector; broadcasting only its first element (the pre-fix behavior) offsets every
   * non-zero class by base_score[k] - base_score[0], so this asserts each group's margin, not just
   * leaf indices or the argmax.
   */
  @Test
  public void reproducesMultinomialMarginsAndProbabilities() throws Exception {
    Random random = new Random(15L);
    DMatrix dataset = BoosterTestUtils.multinomialDataset();
    Booster booster = BoosterTestUtils.trainMultinomialBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int i = 0; i < 30; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      FVec fvec = FVec.fromArray(row);

      assertArrayEquals(
          "margin mismatch on row " + i,
          nativeMargins(booster, row, NUM_FEATURES),
          toDoubles(predictor.predictRaw(fvec)),
          EPS);
      assertArrayEquals(
          "probability mismatch on row " + i,
          nativeProbabilities(booster, row, NUM_FEATURES),
          toDoubles(predictor.predict(fvec)),
          EPS);
    }
  }

  @Test
  public void reproducesBinaryProbability() throws Exception {
    Random random = new Random(13L);
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster booster = BoosterTestUtils.trainBinaryBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int i = 0; i < 30; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      double expected = nativePredict(booster, row, NUM_FEATURES);
      double actual = predictor.predict(FVec.fromArray(row), 0)[0];
      assertEquals("probability mismatch on row " + i, expected, actual, EPS);
    }
  }

  @Test
  public void reproducesGammaMargin() throws Exception {
    Random random = new Random(14L);
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster booster = BoosterTestUtils.trainGammaBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int i = 0; i < 30; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      double expected = nativePredict(booster, row, NUM_FEATURES);
      double actual = predictor.predict(FVec.fromArray(row), 0)[0];
      assertEquals("gamma mismatch on row " + i, expected, actual, EPS);
    }
  }

  /** One row per category so both in-set (go-right) and out-of-set (go-left) branches are hit. */
  @Test
  public void reproducesCategoricalSplits() throws Exception {
    DMatrix dataset = BoosterTestUtils.categoricalDataset();
    Booster booster = BoosterTestUtils.trainCategoricalBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int cat = 0; cat < 4; cat++) {
      float[] row = {(cat % 7) * 0.1f, (cat % 5) * 0.2f, cat};

      double expected = nativePredict(booster, row, CATEGORICAL_NUM_FEATURES);
      double actual = predictor.predict(FVec.fromArray(row), 0)[0];
      assertEquals("categorical mismatch on cat " + cat, expected, actual, EPS);

      int[] boosterLeaves = toInts(nativeLeaf(booster, row, CATEGORICAL_NUM_FEATURES));
      int[] predictorLeaves = predictor.predictLeaf(FVec.fromArray(row));
      assertArrayEquals("categorical leaf mismatch on cat " + cat, boosterLeaves, predictorLeaves);
    }
  }

  /**
   * Deep ({@code max_depth} 8), 200-tree numeric parity. Small fixtures never stress {@link
   * com.yelp.xgboost.tree.PreorderRegTree}'s primitive-array repacking; a deep many-tree model
   * does, so an index-arithmetic bug in the repack would surface here as a margin divergence.
   */
  @Test
  public void reproducesDeepManyTreeModel() throws Exception {
    Random random = new Random(50L);
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster booster = BoosterTestUtils.trainDeepBinaryBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int i = 0; i < 50; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      FVec fvec = FVec.fromArray(row);
      assertEquals(
          "deep-model margin mismatch on row " + i,
          nativeMargins(booster, row, NUM_FEATURES)[0],
          predictor.predictRaw(fvec)[0],
          EPS);
      assertArrayEquals(
          "deep-model leaf mismatch on row " + i,
          toInts(nativeLeaf(booster, row, NUM_FEATURES)),
          predictor.predictLeaf(fvec));
    }
  }

  /**
   * Categorical splits with cardinality 50, spanning past the 32-category bitfield word boundary.
   * One row per category exercises both words of every split's category set, plus a NaN category
   * that must route the split default. Pins {@link com.yelp.xgboost.tree.CategoricalRegTree}
   * against native for the high-cardinality case the 4-category test cannot reach.
   */
  @Test
  public void reproducesHighCardinalityCategoricalSplits() throws Exception {
    DMatrix dataset = BoosterTestUtils.highCardinalityCategoricalDataset();
    Booster booster = BoosterTestUtils.trainHighCardinalityCategoricalBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    int numFeatures = BoosterTestUtils.HIGH_CARD_NUM_FEATURES;
    for (int cat = 0; cat < BoosterTestUtils.HIGH_CARD_CATEGORIES; cat++) {
      float[] row = {(cat % 11) * 0.05f, cat};

      double expected = nativePredict(booster, row, numFeatures);
      double actual = predictor.predict(FVec.fromArray(row), 0)[0];
      assertEquals("high-cardinality mismatch on cat " + cat, expected, actual, EPS);

      assertArrayEquals(
          "high-cardinality leaf mismatch on cat " + cat,
          toInts(nativeLeaf(booster, row, numFeatures)),
          predictor.predictLeaf(FVec.fromArray(row)));
    }

    float[] nanCategoryRow = {0.3f, Float.NaN};
    assertEquals(
        "a NaN categorical value must route the split default exactly as native",
        nativePredict(booster, nanCategoryRow, numFeatures),
        predictor.predict(FVec.fromArray(nanCategoryRow), 0)[0],
        EPS);
  }

  /**
   * End-to-end parity for objectives whose prediction transform was previously only unit-tested (or
   * untested) at the {@link com.yelp.xgboost.learner.ObjFunction} level. Training a real model and
   * matching native pins that the registry entry, base_score link, and transform compose correctly:
   * count:poisson (exp), binary:logitraw / reg:squaredlogerror / rank:pairwise / rank:map
   * (identity), and binary:hinge (0/1 threshold).
   */
  @Test
  public void reproducesUndercoveredObjectives() throws Exception {
    String[] objectives = {
      "count:poisson",
      "binary:logitraw",
      "reg:squaredlogerror",
      "rank:pairwise",
      "rank:map",
      "binary:hinge"
    };
    for (String objective : objectives) {
      Random random = new Random(60L);
      DMatrix dataset = BoosterTestUtils.binomialDataset();
      Booster booster = BoosterTestUtils.trainBoosterWithObjective(dataset, objective);
      Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

      for (int i = 0; i < 20; i++) {
        float[] row = randomRow(random, NUM_FEATURES);
        double expected = nativePredict(booster, row, NUM_FEATURES);
        double actual = predictor.predict(FVec.fromArray(row), 0)[0];
        assertEquals(objective + " mismatch on row " + i, expected, actual, EPS);
      }
    }
  }

  /**
   * A single-tree model (one boosting round) and a stump (one round, splits suppressed by a
   * prohibitive {@code gamma} so every tree collapses to a root-only leaf). Both are degenerate
   * shapes the multi-tree fixtures never hit: the stump must return its root leaf without ever
   * taking a branch, and predictLeaf must report a single index. Pinned against native.
   */
  @Test
  public void reproducesSingleTreeAndStumpModels() throws Exception {
    Random random = new Random(70L);
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster singleTree = BoosterTestUtils.trainSingleTreeBooster(dataset);
    Booster stump = BoosterTestUtils.trainStumpBooster(dataset);
    Predictor singleTreePredictor = BoosterTestUtils.predictorFromBooster(singleTree);
    Predictor stumpPredictor = BoosterTestUtils.predictorFromBooster(stump);

    for (int i = 0; i < 30; i++) {
      float[] row = randomRow(random, NUM_FEATURES);
      FVec fvec = FVec.fromArray(row);

      assertEquals(
          "single-tree probability mismatch on row " + i,
          nativePredict(singleTree, row, NUM_FEATURES),
          singleTreePredictor.predict(fvec)[0],
          EPS);
      assertArrayEquals(
          "single-tree leaf mismatch on row " + i,
          toInts(nativeLeaf(singleTree, row, NUM_FEATURES)),
          singleTreePredictor.predictLeaf(fvec));

      assertEquals(
          "stump probability mismatch on row " + i,
          nativePredict(stump, row, NUM_FEATURES),
          stumpPredictor.predict(fvec)[0],
          EPS);
      int[] stumpLeaves = stumpPredictor.predictLeaf(fvec);
      assertEquals("a stump has exactly one tree", 1, stumpLeaves.length);
      assertArrayEquals(
          "stump leaf mismatch on row " + i,
          toInts(nativeLeaf(stump, row, NUM_FEATURES)),
          stumpLeaves);
    }
  }

  /**
   * The prediction transforms mutate their input array in place, so a returned buffer must never be
   * shared across calls. Two predictions on different rows, and a margin call followed by a
   * probability call on the same row, must each return independent arrays whose values do not
   * alias.
   */
  @Test
  public void predictReturnsIndependentArraysAcrossCalls() throws Exception {
    DMatrix dataset = BoosterTestUtils.multinomialDataset();
    Booster booster = BoosterTestUtils.trainMultinomialBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    FVec rowA = FVec.fromArray(randomRow(new Random(80L), NUM_FEATURES));
    FVec rowB = FVec.fromArray(randomRow(new Random(81L), NUM_FEATURES));

    float[] probsA = predictor.predict(rowA);
    float[] snapshotA = probsA.clone();
    float[] probsB = predictor.predict(rowB);

    org.junit.Assert.assertNotSame("each predict call must allocate its own array", probsA, probsB);
    assertArrayEquals(
        "a later predict call must not mutate an earlier returned array", snapshotA, probsA, 0.0f);

    float[] margin = predictor.predictRaw(rowA);
    float[] marginSnapshot = margin.clone();
    float[] probs = predictor.predict(rowA);
    org.junit.Assert.assertNotSame("margin and probability arrays must be distinct", margin, probs);
    assertArrayEquals(
        "a probability call must not mutate a prior margin array", marginSnapshot, margin, 0.0f);
  }

  /**
   * A base row of zeros with a single NaN in each feature position in turn. A NaN must route the
   * split default; a fall-through {@code NaN < split_cond} comparison would route it the other way.
   */
  @Test
  public void routesDenseNaNToSplitDefault() throws Exception {
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster booster = BoosterTestUtils.trainRegressionBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int missingIdx = 0; missingIdx < NUM_FEATURES; missingIdx++) {
      float[] row = new float[NUM_FEATURES];
      row[missingIdx] = Float.NaN;

      double expected = nativePredict(booster, row, NUM_FEATURES);
      double actual = predictor.predict(FVec.fromArray(row), 0)[0];
      assertEquals("NaN-at-" + missingIdx + " mismatch", expected, actual, EPS);
    }
  }

  /** Keep a single present feature and omit the rest; omitted indices are missing to xgboost. */
  @Test
  public void routesOmittedSparseIndexToSplitDefault() throws Exception {
    DMatrix dataset = BoosterTestUtils.binomialDataset();
    Booster booster = BoosterTestUtils.trainRegressionBooster(dataset);
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    for (int presentIdx = 0; presentIdx < NUM_FEATURES; presentIdx++) {
      int[] indices = {presentIdx};
      float[] values = {1.0f};

      double expected = nativeSparsePredict(booster, indices, values, NUM_FEATURES);

      Map<Integer, Float> sparse = new HashMap<>();
      sparse.put(presentIdx, 1.0f);
      double actual = predictor.predict(FVec.fromMap(sparse), 0)[0];
      assertEquals("sparse-present-" + presentIdx + " mismatch", expected, actual, EPS);
    }
  }

  /**
   * Regression guard for the ai.h2o predictor dead-end: its ModelReader read UBJSON bytes as an
   * 8-byte length prefix and threw {@code IOException("Too long string")}. The reader must not.
   */
  @Test
  public void loadingUbjsonModelDoesNotThrowTooLongString() throws Exception {
    Booster binary = BoosterTestUtils.trainBinaryBooster(BoosterTestUtils.binomialDataset());
    Booster multinomial =
        BoosterTestUtils.trainMultinomialBooster(BoosterTestUtils.multinomialDataset());
    BoosterTestUtils.predictorFromBooster(binary);
    BoosterTestUtils.predictorFromBooster(multinomial);
  }

  private static float[] randomRow(Random random, int numFeatures) {
    float[] row = new float[numFeatures];
    for (int j = 0; j < numFeatures; j++) {
      row[j] = random.nextFloat();
    }
    return row;
  }

  private static double nativePredict(Booster booster, float[] row, int numFeatures)
      throws Exception {
    DMatrix dm = new DMatrix(row, 1, numFeatures, Float.NaN);
    return booster.predict(dm, false, 0)[0][0];
  }

  private static float[] nativeLeaf(Booster booster, float[] row, int numFeatures)
      throws Exception {
    DMatrix dm = new DMatrix(row, 1, numFeatures, Float.NaN);
    return booster.predictLeaf(dm, 0)[0];
  }

  private static double[] nativeMargins(Booster booster, float[] row, int numFeatures)
      throws Exception {
    DMatrix dm = new DMatrix(row, 1, numFeatures, Float.NaN);
    return toDoubles(booster.predict(dm, true, 0)[0]);
  }

  private static double[] nativeProbabilities(Booster booster, float[] row, int numFeatures)
      throws Exception {
    DMatrix dm = new DMatrix(row, 1, numFeatures, Float.NaN);
    return toDoubles(booster.predict(dm, false, 0)[0]);
  }

  private static double[] toDoubles(float[] values) {
    double[] out = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = values[i];
    }
    return out;
  }

  private static double nativeSparsePredict(
      Booster booster, int[] indices, float[] values, int size) throws Exception {
    DMatrix dm =
        new DMatrix(
            java.util.List.of(new LabeledPoint(0.0f, size, indices, values)).iterator(), null);
    return booster.predict(dm, false, 0)[0][0];
  }

  private static int[] toInts(float[] values) {
    int[] out = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = (int) values[i];
    }
    return out;
  }
}
