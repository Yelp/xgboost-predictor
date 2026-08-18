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

import static com.yelp.xgboost.testing.BoosterTestUtils.NUM_FEATURES;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.yelp.xgboost.testing.BoosterTestUtils;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import org.junit.Test;

/**
 * Covers public {@link Predictor} entry points and {@link FVec} behaviors not exercised by the
 * parity suites: single-value {@code predictSingle}, {@code ntree_limit} truncation, the
 * treat-zero-as-missing dense path, edge inputs (all-missing row, out-of-range index, present
 * non-finite values), sparse-vs-dense representation equivalence, and the shared-instance
 * concurrency contract that online serving relies on.
 */
public class PredictorApiTest {

  private static final double EPS = 1e-5;

  @Test
  public void predictSingleMatchesArrayPredict() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(31L);
    for (int i = 0; i < 20; i++) {
      FVec fvec = FVec.fromArray(randomRow(random));
      assertEquals(
          "predictSingle must match predict()[0] on row " + i,
          predictor.predict(fvec)[0],
          predictor.predictSingle(fvec),
          EPS);
      assertEquals(
          "predictSingle margin must match predict(margin)[0] on row " + i,
          predictor.predictRaw(fvec)[0],
          predictor.predictSingleRaw(fvec),
          EPS);
    }
  }

  @Test
  public void ntreeLimitTruncatesTreeContribution() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    FVec fvec = FVec.fromArray(randomRow(new Random(32L)));

    float firstTreeOnly = predictor.predictRaw(fvec, 1)[0];
    float allTrees = predictor.predictRaw(fvec)[0];
    assertNotEquals(
        "15-tree margin should differ from a single-tree margin", allTrees, firstTreeOnly);

    assertEquals(
        "ntree_limit=1 must agree with the native Booster limited to one tree",
        nativeMarginLimited(booster, fvec, 1),
        firstTreeOnly,
        EPS);
  }

  /**
   * {@code ntree_limit} on a multiclass model is the subtle case: {@link
   * com.yelp.xgboost.gbm.GBTree#pred} applies the limit per output group while {@code predPath}
   * (leaf) applies it to the flat tree array. Sweeping several limits against the native Booster
   * pins both interpretations up to and including {@code limit == rounds}. The limit is in units of
   * boosting rounds, so 15 (the trained round count) is the maximum the native API accepts.
   */
  @Test
  public void ntreeLimitSweepMatchesNativeForMultinomial() throws Exception {
    Booster booster =
        BoosterTestUtils.trainMultinomialBooster(BoosterTestUtils.multinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(40L);
    for (int i = 0; i < 20; i++) {
      float[] row = randomRow(random);
      FVec fvec = FVec.fromArray(row);
      for (int limit : new int[] {1, 2, 5, 15}) {
        DMatrix dm = new DMatrix(row, 1, NUM_FEATURES, Float.NaN);
        assertArrayEquals(
            "multiclass margin mismatch at ntree_limit=" + limit + " on row " + i,
            toDoubles(booster.predict(dm, true, limit)[0]),
            toDoubles(predictor.predictRaw(fvec, limit)),
            EPS);
        assertArrayEquals(
            "multiclass probability mismatch at ntree_limit=" + limit + " on row " + i,
            toDoubles(booster.predict(dm, false, limit)[0]),
            toDoubles(predictor.predict(fvec, limit)),
            EPS);
      }
    }
  }

  /**
   * An {@code ntree_limit} outside the valid range must throw a clean typed exception, matching
   * native xgboost ("Out of range for tree layers") rather than reading past the tree array or
   * silently clamping. Both a limit past the trained tree count and a negative limit are rejected,
   * on every gbtree entry point that takes a limit.
   */
  @Test
  public void ntreeLimitOutOfRangeThrowsIllegalArgument() throws Exception {
    Booster booster = BoosterTestUtils.trainBinaryBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);
    FVec fvec = FVec.fromArray(randomRow(new Random(43L)));

    for (int limit : new int[] {1000, -1}) {
      assertThrows(IllegalArgumentException.class, () -> predictor.predictRaw(fvec, limit));
      assertThrows(IllegalArgumentException.class, () -> predictor.predictSingleRaw(fvec, limit));
      assertThrows(IllegalArgumentException.class, () -> predictor.predictLeaf(fvec, limit));
    }
  }

  /** {@code predictLeaf} with a non-zero limit is otherwise only ever invoked with 0. */
  @Test
  public void ntreeLimitOnPredictLeafMatchesNative() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(41L);
    for (int i = 0; i < 20; i++) {
      float[] row = randomRow(random);
      FVec fvec = FVec.fromArray(row);
      for (int limit : new int[] {1, 3, 15}) {
        DMatrix dm = new DMatrix(row, 1, NUM_FEATURES, Float.NaN);
        int[] expected = toInts(booster.predictLeaf(dm, limit)[0]);
        assertArrayEquals(
            "leaf indices mismatch at ntree_limit=" + limit + " on row " + i,
            expected,
            predictor.predictLeaf(fvec, limit));
      }
    }
  }

  /** {@code predictSingle} at a non-zero limit is untested elsewhere. */
  @Test
  public void ntreeLimitOnPredictSingleMatchesArrayPredict() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(42L);
    for (int i = 0; i < 20; i++) {
      FVec fvec = FVec.fromArray(randomRow(random));
      for (int limit : new int[] {1, 3, 15}) {
        assertEquals(
            "predictSingle must match predict()[0] at ntree_limit=" + limit + " on row " + i,
            predictor.predictRaw(fvec, limit)[0],
            predictor.predictSingleRaw(fvec, limit),
            0.0f);
      }
    }
  }

  @Test
  public void treatZeroAsMissingRoutesZeroToSplitDefault() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Map<Integer, Float> present = new HashMap<>();
    present.put(0, 1.0f);
    float sparseMargin = predictor.predictRaw(FVec.fromMap(present), 0)[0];

    float[] denseZeros = new float[NUM_FEATURES];
    denseZeros[0] = 1.0f;
    float zeroAsMissingMargin =
        predictor.predictRaw(FVec.fromArrayWithZeroAsMissing(denseZeros), 0)[0];

    assertEquals(
        "a dense row with treatsZeroAsNA must match the equivalent sparse row",
        sparseMargin,
        zeroAsMissingMargin,
        EPS);
  }

  @Test
  public void allMissingRowPredictsBaseScorePath() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    float[] allNaN = new float[NUM_FEATURES];
    java.util.Arrays.fill(allNaN, Float.NaN);
    FVec fvec = FVec.fromArray(allNaN);

    float actual = predictor.predictRaw(fvec)[0];
    assertEquals(
        "all-missing row must route every split default and match native",
        nativeMargin(booster, allNaN),
        actual,
        EPS);
  }

  @Test
  public void outOfRangeFeatureIndexIsTreatedAsMissing() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Map<Integer, Float> sparse = new HashMap<>();
    sparse.put(0, 0.7f);
    sparse.put(999, 42.0f);
    FVec withOutOfRange = FVec.fromMap(sparse);

    Map<Integer, Float> onlyValid = new HashMap<>();
    onlyValid.put(0, 0.7f);
    FVec withoutOutOfRange = FVec.fromMap(onlyValid);

    assertEquals(
        "an out-of-range feature index must not affect the prediction",
        predictor.predictRaw(withoutOutOfRange)[0],
        predictor.predictRaw(withOutOfRange)[0],
        0.0);
  }

  @Test
  public void multiSoftmaxOnUbjsonPathReturnsClassIndex() throws Exception {
    Booster booster =
        BoosterTestUtils.trainMultinomialSoftmaxBooster(BoosterTestUtils.multinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(33L);
    for (int i = 0; i < 20; i++) {
      float[] row = randomRow(random);
      float predicted = predictor.predict(FVec.fromArray(row), 0)[0];
      assertEquals(
          "softmax must return an integral class index", predicted, Math.rint(predicted), 0.0);
      assertTrue("class index in range", predicted >= 0 && predicted <= 2);
      assertEquals(
          "softmax class must match native argmax on row " + i,
          nativePredict(booster, row),
          predicted,
          0.0);
    }
  }

  @Test
  public void convenienceOverloadsMatchTheirExplicitForms() throws Exception {
    Booster booster = BoosterTestUtils.trainRegressionBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);
    FVec fvec = FVec.fromArray(randomRow(new Random(34L)));

    assertEquals(predictor.predict(fvec, 0)[0], predictor.predict(fvec)[0], 0.0);
    assertEquals(predictor.predictRaw(fvec, 0)[0], predictor.predictRaw(fvec)[0], 0.0);
    assertEquals(predictor.predictSingle(fvec, 0), predictor.predictSingle(fvec), 0.0f);
    assertEquals(predictor.predictSingleRaw(fvec, 0), predictor.predictSingleRaw(fvec), 0.0f);

    float[] raw = predictor.predictRaw(fvec);
    assertArrayEquals(predictor.predict(fvec), predictor.predTransform(raw), 0.0f);
  }

  @Test
  public void baseScoreIsExposedAsAFiniteMargin() throws Exception {
    Booster booster = BoosterTestUtils.trainBinaryBooster(BoosterTestUtils.binomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    assertTrue(
        "getBaseScore must expose a finite margin-space intercept",
        Float.isFinite(predictor.getBaseScore()));
  }

  @Test
  public void predictSingleRejectsMultiOutputModel() throws Exception {
    Booster booster =
        BoosterTestUtils.trainMultinomialBooster(BoosterTestUtils.multinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);
    FVec fvec = FVec.fromArray(randomRow(new Random(35L)));

    assertThrows(IllegalStateException.class, () -> predictor.predictSingle(fvec));
  }

  @Test
  public void nanFeatureValuesMatchNativeMissingRouting() throws Exception {
    Booster booster =
        BoosterTestUtils.trainSparseBinaryBooster(BoosterTestUtils.sparseBinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(36L);
    int changedByNaN = 0;
    for (int i = 0; i < 200; i++) {
      float[] row = randomRow(random);
      float[] dense = row.clone();
      for (int k = 0; k < 4; k++) {
        row[(i + k * 3) % NUM_FEATURES] = Float.NaN;
      }

      float actual = predictor.predictRaw(FVec.fromArray(row), 0)[0];
      assertEquals(
          "a present NaN must route the split default exactly as native (missing=NaN) on row " + i,
          nativeMargin(booster, row),
          actual,
          EPS);

      if (Math.abs(actual - predictor.predictRaw(FVec.fromArray(dense), 0)[0]) > EPS) {
        changedByNaN++;
      }
    }
    assertTrue(
        "sparse-trained model must have splits whose NaN routing differs from the dense row, "
            + "otherwise this test would pass vacuously",
        changedByNaN > 0);
  }

  @Test
  public void sparseAndDenseRepresentationsAgreeAndMatchNative() throws Exception {
    Booster booster =
        BoosterTestUtils.trainSparseBinaryBooster(BoosterTestUtils.sparseBinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(39L);
    for (int i = 0; i < 200; i++) {
      float[] dense = randomRow(random);
      for (int k = 0; k < 3; k++) {
        dense[(i + k * 4) % NUM_FEATURES] = 0.0f;
      }

      Map<Integer, Float> full = new HashMap<>();
      for (int j = 0; j < NUM_FEATURES; j++) {
        full.put(j, dense[j]);
      }

      float denseMargin = predictor.predictRaw(FVec.fromArray(dense), 0)[0];
      float sparseMargin = predictor.predictRaw(FVec.fromMap(full), 0)[0];
      float nativeMargin = nativeMargin(booster, dense);

      assertEquals(
          "dense array and full sparse map must agree on row " + i,
          denseMargin,
          sparseMargin,
          0.0f);
      assertEquals(
          "a present 0.0 must not be treated as missing (dense path) on row " + i,
          nativeMargin,
          denseMargin,
          EPS);
      assertEquals(
          "a present 0.0 must not be treated as missing (map path) on row " + i,
          nativeMargin,
          sparseMargin,
          EPS);
    }
  }

  @Test
  public void presentNanRoutesIdenticallyToAnAbsentFeature() throws Exception {
    Booster booster =
        BoosterTestUtils.trainSparseBinaryBooster(BoosterTestUtils.sparseBinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    Random random = new Random(38L);
    for (int i = 0; i < 50; i++) {
      float[] row = randomRow(random);
      Map<Integer, Float> present = new HashMap<>();
      for (int j = 0; j < NUM_FEATURES; j++) {
        present.put(j, row[j]);
      }
      for (int k = 0; k < 4; k++) {
        int missingIndex = (i + k * 3) % NUM_FEATURES;
        row[missingIndex] = Float.NaN;
        present.remove(missingIndex);
      }

      Map<Integer, Float> withNanEntries = new HashMap<>(present);
      for (int k = 0; k < 4; k++) {
        withNanEntries.put((i + k * 3) % NUM_FEATURES, Float.NaN);
      }

      float nanAsPresent = predictor.predictRaw(FVec.fromArray(row), 0)[0];
      float trulyAbsent = predictor.predictRaw(FVec.fromMap(present), 0)[0];
      float nanInMap = predictor.predictRaw(FVec.fromMap(withNanEntries), 0)[0];
      assertEquals(
          "a present NaN (array path) must route identically to an absent key (map path) on row "
              + i,
          trulyAbsent,
          nanAsPresent,
          0.0f);
      assertEquals(
          "an explicit NaN map entry must route identically to an absent key on row " + i,
          trulyAbsent,
          nanInMap,
          0.0f);
    }
  }

  @Test
  public void oneLoadedPredictorIsSafeToShareAcrossThreads() throws Exception {
    Booster booster =
        BoosterTestUtils.trainMultinomialBooster(BoosterTestUtils.multinomialDataset());
    Predictor predictor = BoosterTestUtils.predictorFromBooster(booster);

    int rows = 64;
    Random random = new Random(37L);
    List<FVec> inputs = new ArrayList<>();
    List<float[]> expected = new ArrayList<>();
    for (int i = 0; i < rows; i++) {
      FVec fvec = FVec.fromArray(randomRow(random));
      inputs.add(fvec);
      expected.add(predictor.predictRaw(fvec));
    }

    int threads = 8;
    int iterationsPerThread = 500;
    ExecutorService pool = Executors.newFixedThreadPool(threads);
    try {
      List<Callable<Void>> tasks = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        tasks.add(
            () -> {
              for (int it = 0; it < iterationsPerThread; it++) {
                int r = it % rows;
                assertArrayEquals(
                    "concurrent prediction must equal the single-threaded result",
                    expected.get(r),
                    predictor.predictRaw(inputs.get(r), 0),
                    0.0f);
              }
              return null;
            });
      }
      for (Future<Void> future : pool.invokeAll(tasks)) {
        future.get();
      }
    } finally {
      pool.shutdownNow();
    }
  }

  private static float[] randomRow(Random random) {
    float[] row = new float[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) {
      row[j] = random.nextFloat();
    }
    return row;
  }

  private static float nativeMargin(Booster booster, float[] row) throws Exception {
    DMatrix dm = new DMatrix(row, 1, NUM_FEATURES, Float.NaN);
    return booster.predict(dm, true, 0)[0][0];
  }

  private static float nativeMarginLimited(Booster booster, FVec fvec, int ntreeLimit)
      throws Exception {
    float[] row = new float[NUM_FEATURES];
    for (int j = 0; j < NUM_FEATURES; j++) {
      Float v = fvec.fvalue(j);
      row[j] = v == null ? Float.NaN : v;
    }
    DMatrix dm = new DMatrix(row, 1, NUM_FEATURES, Float.NaN);
    return booster.predict(dm, true, ntreeLimit)[0][0];
  }

  private static float nativePredict(Booster booster, float[] row) throws Exception {
    DMatrix dm = new DMatrix(row, 1, NUM_FEATURES, Float.NaN);
    return booster.predict(dm, false, 0)[0][0];
  }

  private static double[] toDoubles(float[] values) {
    double[] out = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = values[i];
    }
    return out;
  }

  private static int[] toInts(float[] values) {
    int[] out = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = (int) values[i];
    }
    return out;
  }
}
