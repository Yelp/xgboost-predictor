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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import com.yelp.xgboost.FVec;
import com.yelp.xgboost.Predictor;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Zero-regression guarantee for the legacy-binary {@code ModelReader} path.
 *
 * <p>The 8 models under {@code datasources/golden/} are legacy pre-1.0 binary format (leading byte
 * {@code 0x62}, "binf"), captured once on xgboost 2.0.3. Native xgboost 3.3.0 can no longer read or
 * emit that format, so these fixtures are the ONLY coverage of the legacy reader and their expected
 * outputs are frozen literals in {@code v2.0.3/golden.json}, never recomputed from a live model.
 *
 * <p>The fixture also embeds, under {@code datasets}, the exact feature vectors for each row index
 * (dense arrays for the binomial agaricus dataset, sparse index/value maps for the multinomial iris
 * dataset). Embedding them keeps this test self-contained and replicable: no Spark, no libsvm/CSV
 * loading, just the models plus the JSON. Every objective's frozen predictions, margins,
 * probabilities and leaf indices are asserted exactly.
 */
public class GoldenValueParityTest {

  private static final String GOLDEN_JSON = "datasources/golden/v2.0.3/golden.json";
  private static final double EPS = 1e-6;

  @Test
  public void reproducesFrozenLegacyBinaryOutputs() throws Exception {
    Json.Obj root = Json.parse(readResource(GOLDEN_JSON)).asObj();
    assertEquals(
        "golden baseline must stay pinned to its captured xgboost version",
        "2.0.3",
        root.get("xgboostVersion").asString());
    Map<String, Map<Integer, FVec>> datasets = loadDatasets(root.get("datasets").asObj());

    List<Json> cases = root.get("cases").asArr();
    assertTrue("Expected golden cases", !cases.isEmpty());

    for (Json caseValue : cases) {
      Json.Obj c = caseValue.asObj();
      String modelResource = c.get("modelResource").asString();
      String dataset = c.get("dataset").asString();
      int numClasses = (int) c.get("numClasses").asDouble();
      boolean hasProbabilities = c.get("probabilities").asArr().size() > 0;

      Predictor predictor = predictorFor(modelResource);
      Map<Integer, FVec> byRow = datasets.get(dataset);
      assertNotNull("Unknown dataset " + dataset, byRow);

      int[] rowIndices = intArray(c.get("rowIndices").asArr());
      double[] predictions = doubleArray(c.get("predictions").asArr());
      double[][] margins = doubleMatrix(c.get("margins").asArr());
      double[][] probabilities = doubleMatrix(c.get("probabilities").asArr());
      int[][] leafIndices = intMatrix(c.get("leafIndices").asArr());

      for (int i = 0; i < rowIndices.length; i++) {
        FVec fvec = byRow.get(rowIndices[i]);
        assertNotNull("Missing feature vector for row " + rowIndices[i], fvec);
        String where = modelResource + " row " + rowIndices[i];

        float[] raw = predictor.predict(fvec);
        double prediction = (numClasses > 2 && hasProbabilities) ? argmax(raw) : (double) raw[0];
        assertEquals("prediction " + where, predictions[i], prediction, EPS);

        float[] margin = predictor.predictRaw(fvec);
        assertArrayEquals("margin " + where, margins[i], toDoubles(margin), EPS);

        if (hasProbabilities) {
          double[] probs = (numClasses == 2) ? new double[] {1.0 - raw[0], raw[0]} : toDoubles(raw);
          assertArrayEquals("probability " + where, probabilities[i], probs, EPS);
        }

        int[] leaves = predictor.predictLeaf(fvec);
        assertArrayEquals("leaf " + where, leafIndices[i], leaves);
      }
    }
  }

  private static Predictor predictorFor(String modelResource) throws Exception {
    try (InputStream model =
        GoldenValueParityTest.class.getClassLoader().getResourceAsStream(modelResource)) {
      assertNotNull("Missing golden model " + modelResource, model);
      return PredictorFactory.fromModelStream(model);
    }
  }

  private static Map<String, Map<Integer, FVec>> loadDatasets(Json.Obj datasets) {
    Map<String, Map<Integer, FVec>> out = new HashMap<>();
    out.put("binomial", denseRows(datasets.get("binomial").asObj()));
    out.put("multinomial", sparseRows(datasets.get("multinomial").asObj()));
    return out;
  }

  private static Map<Integer, FVec> denseRows(Json.Obj rows) {
    Map<Integer, FVec> out = new HashMap<>();
    for (Map.Entry<String, Json> e : rows.fields.entrySet()) {
      out.put(Integer.parseInt(e.getKey()), FVec.fromArray(doubleArray(e.getValue().asArr())));
    }
    return out;
  }

  private static Map<Integer, FVec> sparseRows(Json.Obj rows) {
    Map<Integer, FVec> out = new HashMap<>();
    for (Map.Entry<String, Json> e : rows.fields.entrySet()) {
      Map<Integer, Float> sparse = new HashMap<>();
      for (Map.Entry<String, Json> f : e.getValue().asObj().fields.entrySet()) {
        sparse.put(Integer.parseInt(f.getKey()), (float) f.getValue().asDouble());
      }
      out.put(Integer.parseInt(e.getKey()), FVec.fromMap(sparse));
    }
    return out;
  }

  private static double argmax(float[] values) {
    int best = 0;
    for (int i = 1; i < values.length; i++) {
      if (values[i] > values[best]) {
        best = i;
      }
    }
    return best;
  }

  private static double[] toDoubles(float[] values) {
    double[] out = new double[values.length];
    for (int i = 0; i < values.length; i++) {
      out[i] = values[i];
    }
    return out;
  }

  private static int[] intArray(List<Json> arr) {
    int[] out = new int[arr.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = (int) arr.get(i).asDouble();
    }
    return out;
  }

  private static double[] doubleArray(List<Json> arr) {
    double[] out = new double[arr.size()];
    for (int i = 0; i < out.length; i++) {
      out[i] = arr.get(i).asDouble();
    }
    return out;
  }

  private static double[][] doubleMatrix(List<Json> arr) {
    double[][] out = new double[arr.size()][];
    for (int i = 0; i < out.length; i++) {
      out[i] = doubleArray(arr.get(i).asArr());
    }
    return out;
  }

  private static int[][] intMatrix(List<Json> arr) {
    int[][] out = new int[arr.size()][];
    for (int i = 0; i < out.length; i++) {
      out[i] = intArray(arr.get(i).asArr());
    }
    return out;
  }

  private static String readResource(String name) throws Exception {
    try (InputStream in = GoldenValueParityTest.class.getClassLoader().getResourceAsStream(name)) {
      assertNotNull("Missing fixture " + name, in);
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }
  }
}
