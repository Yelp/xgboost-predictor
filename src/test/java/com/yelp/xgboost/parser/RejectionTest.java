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

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.yelp.xgboost.learner.ObjFunction;
import com.yelp.xgboost.parser.UValue.UArray;
import com.yelp.xgboost.parser.UValue.UObject;
import com.yelp.xgboost.parser.UValue.UString;
import java.io.InputStream;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;

/**
 * Pins the documented scope boundaries: the loader rejects everything outside gbtree numeric/
 * categorical trees rather than silently producing wrong predictions. Model trees are hand-built as
 * {@link UValue} (the format-agnostic tree {@link XGBoostModelParser} consumes), so no exotic
 * booster needs to be trained to reach each rejection branch.
 */
public class RejectionTest {

  @Test
  public void rejectsDartBooster() {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> XGBoostModelParser.parse(model("reg:squarederror", "dart", null)));
    assertTrue(e.getMessage(), e.getMessage().contains("dart"));
  }

  @Test
  public void rejectsGblinearBooster() {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> XGBoostModelParser.parse(model("reg:squarederror", "gblinear", null)));
    assertTrue(e.getMessage(), e.getMessage().contains("gblinear"));
  }

  @Test
  public void rejectsUnknownBooster() {
    assertThrows(
        IllegalArgumentException.class,
        () -> XGBoostModelParser.parse(model("reg:squarederror", "mystery", null)));
  }

  @Test
  public void rejectsVectorLeafTrees() {
    UValue gbtreeModel = gbtreeModelWithVectorLeaf();
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> XGBoostModelParser.parse(model("reg:squarederror", "gbtree", gbtreeModel)));
    assertTrue(e.getMessage(), e.getMessage().contains("size_leaf_vector"));
  }

  @Test
  public void rejectsUnsupportedObjectiveAtParse() {
    assertThrows(
        IllegalArgumentException.class,
        () -> XGBoostModelParser.parse(model("survival:aft", "gbtree", null)));
  }

  @Test
  public void rejectsUnsupportedObjectiveByName() {
    assertThrows(IllegalArgumentException.class, () -> ObjFunction.fromName("reg:absoluteerror"));
  }

  @Test
  public void rejectsEmptyModelBytes() {
    assertThrows(
        IllegalArgumentException.class, () -> PredictorFactory.fromModelBytes(new byte[0]));
  }

  /**
   * A leading '{' routes to the UBJSON reader, which decodes XGBoost's binary codec, not textual
   * JSON. Feeding plain-text JSON must fail fast (the reader expects an int64-framed key, not '"'),
   * never silently mis-parse.
   */
  @Test
  public void rejectsPlainTextJson() {
    byte[] jsonText = "{\"learner\": {}}".getBytes(java.nio.charset.StandardCharsets.UTF_8);
    assertThrows(RuntimeException.class, () -> PredictorFactory.fromModelBytes(jsonText));
  }

  /**
   * A UBJSON model truncated mid-stream must fail fast, never return a silently-wrong predictor.
   * The leading &#123; still routes to the UBJSON reader, which must run off the end of the buffer
   * and throw rather than parse a partial tree.
   */
  @Test
  public void rejectsTruncatedUbjsonModel() throws Exception {
    byte[] full = resourceBytes("datasources/golden/v3.3.0/xgboost.model");
    byte[] truncated = Arrays.copyOf(full, full.length / 2);
    assertThrows(RuntimeException.class, () -> PredictorFactory.fromModelBytes(truncated));
  }

  /**
   * A legacy-binary model (no leading &#123;) truncated mid-stream must also fail fast rather than
   * build a partial predictor. The 1.7.6 golden opens with the "binf" magic, so this exercises the
   * legacy branch, not the UBJSON one.
   */
  @Test
  public void rejectsTruncatedLegacyBinaryModel() throws Exception {
    byte[] full = resourceBytes("datasources/golden/v1.7.6/binary_logistic.model");
    byte[] truncated = Arrays.copyOf(full, full.length / 2);
    assertThrows(RuntimeException.class, () -> PredictorFactory.fromModelBytes(truncated));
  }

  private static byte[] resourceBytes(String name) throws Exception {
    try (InputStream in = RejectionTest.class.getClassLoader().getResourceAsStream(name)) {
      assertNotNull("Missing fixture " + name, in);
      return in.readAllBytes();
    }
  }

  private static UValue model(String objective, String boosterName, UValue gbtreeModel) {
    Map<String, UValue> learnerParam = new LinkedHashMap<>();
    learnerParam.put("base_score", new UString("0.5"));
    learnerParam.put("num_class", new UString("0"));
    learnerParam.put("num_feature", new UString("3"));

    Map<String, UValue> objectiveObj = new LinkedHashMap<>();
    objectiveObj.put("name", new UString(objective));

    Map<String, UValue> booster = new LinkedHashMap<>();
    booster.put("name", new UString(boosterName));
    if (gbtreeModel != null) {
      booster.put("model", gbtreeModel);
    }

    Map<String, UValue> learner = new LinkedHashMap<>();
    learner.put("learner_model_param", new UObject(learnerParam));
    learner.put("objective", new UObject(objectiveObj));
    learner.put("gradient_booster", new UObject(booster));

    Map<String, UValue> root = new LinkedHashMap<>();
    root.put("learner", new UObject(learner));
    return new UObject(root);
  }

  private static UValue gbtreeModelWithVectorLeaf() {
    Map<String, UValue> treeParam = new LinkedHashMap<>();
    treeParam.put("num_nodes", new UString("1"));
    treeParam.put("size_leaf_vector", new UString("2"));

    Map<String, UValue> tree = new LinkedHashMap<>();
    tree.put("tree_param", new UObject(treeParam));

    Map<String, UValue> gbtreeModel = new LinkedHashMap<>();
    gbtreeModel.put("trees", new UArray(List.of(new UObject(tree))));
    gbtreeModel.put("tree_info", new UArray(List.of()));
    return new UObject(gbtreeModel);
  }
}
