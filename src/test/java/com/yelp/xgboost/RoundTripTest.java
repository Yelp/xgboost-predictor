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
import static org.junit.Assert.assertNotNull;

import com.yelp.xgboost.parser.PredictorFactory;
import com.yelp.xgboost.testing.BoosterTestUtils;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import ml.dmlc.xgboost4j.java.Booster;
import org.junit.Test;

/**
 * Pins the lossless re-serialize guarantee: a loaded {@link Predictor} retains the raw model bytes
 * ({@link Predictor#getRawModel()}) so it can be written back and reloaded to an identical model.
 * MLeap's predictor {@code store} op relies on this to round-trip a bundle through save -> load ->
 * predict without re-emitting the parsed tree structures. Covers both the UBJSON path (a freshly
 * trained booster) and the legacy-binary path (a frozen 1.7.6 golden), since {@link
 * PredictorFactory} dispatches the two formats on the first byte and only the UBJSON round-trip was
 * pinned before.
 */
public class RoundTripTest {

  private static final String LEGACY_MODEL = "datasources/golden/v1.7.6/binary_logistic.model";

  @Test
  public void rawModelBytesReloadToIdenticalPredictor() throws Exception {
    Booster booster = BoosterTestUtils.trainBinaryBooster(BoosterTestUtils.binomialDataset());
    Predictor original = BoosterTestUtils.predictorFromBooster(booster);

    byte[] raw = original.getRawModel();
    assertNotNull("loaded predictor must retain its raw model bytes", raw);

    Predictor reloaded = PredictorFactory.fromModelBytes(raw);

    Random random = new Random(21L);
    for (int i = 0; i < 30; i++) {
      float[] row = new float[NUM_FEATURES];
      for (int j = 0; j < NUM_FEATURES; j++) {
        row[j] = random.nextFloat();
      }
      FVec fvec = FVec.fromArray(row);
      assertArrayEquals(
          "prediction diverged after round-trip on row " + i,
          toDoubles(original.predict(fvec)),
          toDoubles(reloaded.predict(fvec)),
          0.0);
      assertArrayEquals(
          "leaf indices diverged after round-trip on row " + i,
          original.predictLeaf(fvec),
          reloaded.predictLeaf(fvec));
    }
  }

  @Test
  public void legacyBinaryRawModelBytesReloadToIdenticalPredictor() throws Exception {
    Predictor original;
    try (InputStream in = getClass().getClassLoader().getResourceAsStream(LEGACY_MODEL)) {
      assertNotNull("Missing legacy golden " + LEGACY_MODEL, in);
      original = PredictorFactory.fromModelStream(in);
    }

    byte[] raw = original.getRawModel();
    assertNotNull("loaded legacy predictor must retain its raw model bytes", raw);

    Predictor reloaded = PredictorFactory.fromModelBytes(raw);

    int numFeatures = 8;
    Random random = new Random(22L);
    for (int i = 0; i < 30; i++) {
      Map<Integer, Float> sparse = new HashMap<>();
      for (int j = 0; j < numFeatures; j++) {
        sparse.put(j, random.nextFloat());
      }
      FVec fvec = FVec.fromMap(sparse);
      assertArrayEquals(
          "legacy prediction diverged after round-trip on row " + i,
          toDoubles(original.predict(fvec)),
          toDoubles(reloaded.predict(fvec)),
          0.0);
      assertArrayEquals(
          "legacy leaf indices diverged after round-trip on row " + i,
          original.predictLeaf(fvec),
          reloaded.predictLeaf(fvec));
    }
  }

  private static double[] toDoubles(float[] values) {
    double[] out = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = values[i];
    }
    return out;
  }
}
