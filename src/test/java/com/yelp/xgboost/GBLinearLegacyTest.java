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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;

import com.yelp.xgboost.parser.PredictorFactory;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;

/**
 * Exercises the {@code gblinear} legacy-binary surface that only the linear booster reaches: {@code
 * predictSingle} agreeing with {@code predict()[0]}, and {@code predictLeaf} throwing since a
 * linear model has no trees. The model is the frozen 1.7.6 gblinear golden.
 *
 * <p>{@code gblinear} is out of the documented gbtree scope for new UBJSON models, but legacy 1.x
 * bundles can carry it, so the loader must handle it deterministically rather than misbehave.
 */
public class GBLinearLegacyTest {

  private static final String GBLINEAR_MODEL = "datasources/golden/v1.7.6/gblinear_logistic.model";
  private static final String GBLINEAR_SOFTPROB_MODEL =
      "datasources/golden/v1.7.6/gblinear_softprob.model";
  private static final double EPS = 1e-6;

  @Test
  public void predictSingleMatchesArrayPredict() throws Exception {
    Predictor predictor = load();
    FVec fvec = row(0.2f, 0.6f, 0.1f, 0.9f);

    assertEquals(
        "gblinear predictSingle must match predict()[0]",
        predictor.predict(fvec)[0],
        predictor.predictSingle(fvec),
        EPS);
    assertEquals(
        "gblinear predictSingle margin must match predict(margin)[0]",
        predictor.predictRaw(fvec)[0],
        predictor.predictSingleRaw(fvec),
        EPS);
  }

  @Test
  public void predictLeafIsUnsupported() throws Exception {
    Predictor predictor = load();
    assertThrows(
        UnsupportedOperationException.class,
        () -> predictor.predictLeaf(row(0.2f, 0.6f, 0.1f, 0.9f)));
  }

  /**
   * A multiclass gblinear model outputs one value per class, so {@code predictSingle} must reject
   * it rather than silently returning the first group. The gbtree rejection path is covered
   * elsewhere; this pins the linear booster's own guard.
   */
  @Test
  public void predictSingleRejectsMultiOutputGblinear() throws Exception {
    Predictor predictor = load(GBLINEAR_SOFTPROB_MODEL);
    assertThrows(
        IllegalStateException.class, () -> predictor.predictSingle(row(0.2f, 0.6f, 0.1f, 0.9f)));
  }

  private static Predictor load() throws Exception {
    return load(GBLINEAR_MODEL);
  }

  private static Predictor load(String resource) throws Exception {
    try (InputStream in = GBLinearLegacyTest.class.getClassLoader().getResourceAsStream(resource)) {
      assertNotNull("Missing gblinear golden " + resource, in);
      return PredictorFactory.fromModelStream(in);
    }
  }

  private static FVec row(float... values) {
    Map<Integer, Float> map = new HashMap<>();
    for (int i = 0; i < values.length; i++) {
      map.put(i, values[i]);
    }
    return FVec.fromMap(map);
  }
}
