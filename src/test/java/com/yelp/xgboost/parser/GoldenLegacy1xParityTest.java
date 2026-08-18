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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/**
 * Zero-regression guarantee for the legacy-binary {@code ModelReader} path across the full span of
 * XGBoost versions Yelp has run in production: 1.0.0 through 1.7.6.
 *
 * <p>That range emits two distinct legacy-binary layouts, both reaching this reader:
 *
 * <ul>
 *   <li>{@code 1.0.0} writes no {@code "binf"} magic and {@code major_version == 1}, exercising the
 *       else-branch of {@code Predictor.ModelParam} (raw base_score, no magic skip).
 *   <li>{@code 1.3.0 .. 1.7.6} prepend the {@code "binf"} magic with {@code major_version == 1},
 *       exercising the magic-skip branch.
 * </ul>
 *
 * <p>The pre-existing {@code v2.0.3/golden.json} covers only the magic branch at {@code
 * major_version == 2}, so neither the no-magic layout nor {@code gblinear} legacy loading was
 * pinned before this test. Each fixture also exercises {@code gblinear} (the linear booster reached
 * through the legacy path, which has no leaf-index support).
 *
 * <p>Fixtures are frozen native outputs, regenerable with {@code generate_legacy_goldens.py} under
 * a matching xgboost venv. Feature vectors are embedded so the test is self-contained.
 */
@RunWith(Parameterized.class)
public class GoldenLegacy1xParityTest {

  private static final double EPS = 1e-6;

  @Parameters(name = "xgboost {0}")
  public static List<String> versions() {
    return Arrays.asList("1.0.0", "1.7.6");
  }

  private final String version;

  public GoldenLegacy1xParityTest(String version) {
    this.version = version;
  }

  @Test
  public void reproducesFrozenLegacyBinaryOutputs() throws Exception {
    String goldenJson = "datasources/golden/v" + version + "/golden.json";
    Json.Obj root = Json.parse(readResource(goldenJson)).asObj();
    assertEquals(
        "golden baseline must stay pinned to its captured xgboost version",
        version,
        root.get("xgboostVersion").asString());

    Map<String, Map<Integer, FVec>> datasets = loadDatasets(root.get("datasets").asObj());

    List<Json> cases = root.get("cases").asArr();
    assertTrue("Expected golden cases", !cases.isEmpty());

    for (Json caseValue : cases) {
      Json.Obj c = caseValue.asObj();
      String modelResource = c.get("modelResource").asString();
      String booster = c.get("booster").asString();
      String objective = c.get("objective").asString();
      int numClasses = (int) c.get("numClasses").asDouble();
      boolean hasProbabilities = c.get("probabilities").asArr().size() > 0;
      boolean hasLeaves = c.get("leafIndices").asArr().size() > 0;

      Predictor predictor = predictorFor(modelResource);
      Map<Integer, FVec> byRow = datasets.get(c.get("dataset").asString());
      assertNotNull("Unknown dataset for " + modelResource, byRow);

      int[] rowIndices = intArray(c.get("rowIndices").asArr());
      double[] predictions = doubleArray(c.get("predictions").asArr());
      double[][] margins = doubleMatrix(c.get("margins").asArr());
      double[][] probabilities = doubleMatrix(c.get("probabilities").asArr());
      int[][] leafIndices = hasLeaves ? intMatrix(c.get("leafIndices").asArr()) : null;

      for (int i = 0; i < rowIndices.length; i++) {
        FVec fvec = byRow.get(rowIndices[i]);
        assertNotNull("Missing feature vector for row " + rowIndices[i], fvec);
        String where = version + " " + modelResource + " row " + rowIndices[i];

        float[] raw = predictor.predict(fvec);
        double prediction = ("multi:softprob".equals(objective)) ? argmax(raw) : (double) raw[0];
        assertEquals("prediction " + where, predictions[i], prediction, EPS);

        float[] margin = predictor.predictRaw(fvec);
        assertArrayEquals("margin " + where, margins[i], toDoubles(margin), EPS);

        if (hasProbabilities) {
          double[] probs = (numClasses == 2) ? new double[] {1.0 - raw[0], raw[0]} : toDoubles(raw);
          assertArrayEquals("probability " + where, probabilities[i], probs, EPS);
        }

        if (hasLeaves) {
          int[] leaves = predictor.predictLeaf(fvec);
          assertArrayEquals("leaf " + where, leafIndices[i], leaves);
        }
      }
    }
  }

  private static Predictor predictorFor(String modelResource) throws Exception {
    try (InputStream model =
        GoldenLegacy1xParityTest.class.getClassLoader().getResourceAsStream(modelResource)) {
      assertNotNull("Missing golden model " + modelResource, model);
      return PredictorFactory.fromModelStream(model);
    }
  }

  private static Map<String, Map<Integer, FVec>> loadDatasets(Json.Obj datasets) {
    Map<String, Map<Integer, FVec>> out = new HashMap<>();
    out.put("binomial", denseRows(datasets.get("binomial").asObj()));
    out.put("multinomial", sparseRows(datasets.get("multinomial").asObj()));
    out.put("positive", denseRows(datasets.get("positive").asObj()));
    out.put("ranking", denseRows(datasets.get("ranking").asObj()));
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
    try (InputStream in =
        GoldenLegacy1xParityTest.class.getClassLoader().getResourceAsStream(name)) {
      assertNotNull("Missing fixture " + name, in);
      return new String(in.readAllBytes(), StandardCharsets.UTF_8);
    }
  }
}
