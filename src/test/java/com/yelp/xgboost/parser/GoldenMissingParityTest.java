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
package com.yelp.xgboost.parser;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.yelp.xgboost.FVec;
import com.yelp.xgboost.Predictor;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Test;

/**
 * End-to-end parity check for the UBJSON parse + predict path against ground-truth fixtures
 * captured from native xgboost (datasources/golden/v3.3.0/, a frozen 3.3.0 UBJSON model). Each row
 * carries its sparse (indices/values) and dense feature representation plus the native prediction
 * for each. The model was trained with missing=0, so the sparse representation (absent = missing)
 * reproduces native_sparse exactly.
 */
public class GoldenMissingParityTest {

  private static final String DIR = "datasources/golden/v3.3.0";
  private static final double EPS = 1e-5;

  private record GoldenRow(
      int[] indices, double[] values, double[] dense, double nativeSparse, double nativeDense) {}

  @Test
  public void reproducesNativeSparsePredictionsFromUbjson() throws Exception {
    Predictor predictor;
    try (InputStream model = resource("xgboost.model")) {
      assertNotNull("Missing golden model " + DIR + "/xgboost.model", model);
      predictor = PredictorFactory.fromModelStream(model);
    }

    List<GoldenRow> rows = loadGolden();
    assertTrue("Expected golden rows", !rows.isEmpty());

    int sparseMatchesNativeSparse = 0;
    int denseMatchesNativeDense = 0;
    for (GoldenRow r : rows) {
      Map<Integer, Float> sparse = new HashMap<>();
      for (int i = 0; i < r.indices.length; i++) {
        sparse.put(r.indices[i], (float) r.values[i]);
      }
      FVec sparseVec = FVec.fromMap(sparse);
      FVec denseVec = FVec.fromArray(r.dense);

      double predSparse = predictor.predict(sparseVec)[0];
      double predDense = predictor.predict(denseVec)[0];

      if (Math.abs(predSparse - r.nativeSparse) < EPS) {
        sparseMatchesNativeSparse++;
      }
      if (Math.abs(predDense - r.nativeDense) < EPS) {
        denseMatchesNativeDense++;
      }
    }

    System.out.println(
        "sparse->native_sparse="
            + sparseMatchesNativeSparse
            + "/"
            + rows.size()
            + " dense->native_dense="
            + denseMatchesNativeDense
            + "/"
            + rows.size());

    assertEquals(
        "sparse input must reproduce native_sparse on every row",
        rows.size(),
        sparseMatchesNativeSparse);
    assertEquals(
        "dense input must reproduce native_dense on every row",
        rows.size(),
        denseMatchesNativeDense);
  }

  private static InputStream resource(String name) {
    return GoldenMissingParityTest.class.getClassLoader().getResourceAsStream(DIR + "/" + name);
  }

  private static List<GoldenRow> loadGolden() throws Exception {
    String json;
    try (InputStream in = resource("golden.json")) {
      assertNotNull("Missing golden fixture " + DIR + "/golden.json", in);
      json = new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }

    List<GoldenRow> rows = new ArrayList<>();
    Matcher rowMatcher = Pattern.compile("(?s)\\{\\s*\"indices\".*?\\}").matcher(json);
    while (rowMatcher.find()) {
      String obj = rowMatcher.group();
      int[] indices = toIntArray(array(obj, "indices"));
      double[] values = array(obj, "values");
      double[] dense = array(obj, "dense");
      double nativeSparse = scalar(obj, "native_sparse");
      double nativeDense = scalar(obj, "native_dense");
      rows.add(new GoldenRow(indices, values, dense, nativeSparse, nativeDense));
    }
    return rows;
  }

  private static double[] array(String obj, String key) {
    Matcher m = Pattern.compile("(?s)\"" + key + "\": \\[([^\\]]*)\\]").matcher(obj);
    if (!m.find()) {
      throw new IllegalStateException("Missing array key: " + key);
    }
    String[] parts = m.group(1).split(",");
    List<Double> out = new ArrayList<>();
    for (String p : parts) {
      String t = p.trim();
      if (!t.isEmpty()) {
        out.add(Double.parseDouble(t));
      }
    }
    double[] result = new double[out.size()];
    for (int i = 0; i < result.length; i++) {
      result[i] = out.get(i);
    }
    return result;
  }

  private static double scalar(String obj, String key) {
    Matcher m = Pattern.compile("\"" + key + "\": (-?[0-9.eE+]+)").matcher(obj);
    if (!m.find()) {
      throw new IllegalStateException("Missing scalar key: " + key);
    }
    return Double.parseDouble(m.group(1));
  }

  private static int[] toIntArray(double[] values) {
    int[] out = new int[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = (int) values[i];
    }
    return out;
  }
}
