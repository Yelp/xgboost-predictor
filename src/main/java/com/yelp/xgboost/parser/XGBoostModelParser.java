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

import com.yelp.xgboost.Predictor;
import com.yelp.xgboost.gbm.GBTree;
import com.yelp.xgboost.gbm.GradBooster;
import com.yelp.xgboost.learner.ObjFunction;
import com.yelp.xgboost.parser.UValue.UNumber;
import com.yelp.xgboost.parser.UValue.UString;
import com.yelp.xgboost.tree.AbstractRegTree;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 * Builds the vendored pure-JVM {@link Predictor} from XGBoost's 3.x JSON/UBJSON model tree.
 *
 * <p>The schema is identical across JSON and UBJSON (same keys, parallel per-node arrays), so this
 * layer is format-agnostic and works on the {@link UValue} tree produced by {@link UBJSONReader}.
 * Field layout follows dmlc/xgboost io_utils.h (tree_field) and gbtree_model.cc, verified against
 * v3.0.0.
 *
 * <p>Scope (v1): gbtree only. dart and gblinear-from-JSON are rejected; vector-leaf / multi-target
 * trees (size_leaf_vector &gt; 1) are rejected. base_score is stored in probability space and
 * converted to margin space via the objective's probToMargin before prediction (see parse).
 */
final class XGBoostModelParser {

  private XGBoostModelParser() {}

  public static Predictor parse(UValue model) {
    UValue learner = model.apply("learner");
    UValue learnerParam = learner.apply("learner_model_param");

    float[] baseScore = parseBaseScores(paramString(learnerParam, "base_score"));
    int numClass = Integer.parseInt(paramString(learnerParam, "num_class"));
    int numFeature = Integer.parseInt(paramString(learnerParam, "num_feature"));

    String objectiveName = learner.apply("objective").apply("name").str();
    ObjFunction obj = ObjFunction.fromName(objectiveName);

    UValue booster = learner.apply("gradient_booster");
    String boosterName = booster.apply("name").str();
    rejectUnsupportedBooster(boosterName);

    GradBooster gbm = buildGBTree(booster.apply("model"), numClass, numFeature);

    // learner_model_param.base_score is serialized in probability space (learner.cc ToJson writes
    // the legacy user-facing value, not the margin-converted one). XGBoost applies ProbToMargin at
    // load, matching the legacy Predictor's major_version >= 1 path, so convert here before
    // predicting. Multiclass models store one intercept per output group, so convert each.
    float[] baseMargins = new float[baseScore.length];
    for (int i = 0; i < baseScore.length; i++) {
      baseMargins[i] = obj.probToMargin(baseScore[i]);
    }
    return new Predictor(obj, gbm, baseMargins);
  }

  private static void rejectUnsupportedBooster(String name) {
    switch (name) {
      case "gbtree":
        return;
      case "dart":
        throw new IllegalArgumentException(
            "The 'dart' booster is not supported by this XGBoost predictor. "
                + "Only 'gbtree' models can be served.");
      default:
        throw new IllegalArgumentException(
            "Unsupported gradient booster: '" + name + "'. Only 'gbtree' is supported.");
    }
  }

  private static GradBooster buildGBTree(UValue gbtreeModel, int numClass, int numFeature) {
    List<UValue> treeValues = gbtreeModel.apply("trees").arr();
    AbstractRegTree.Param[] treeParams = new AbstractRegTree.Param[treeValues.size()];
    for (int i = 0; i < treeParams.length; i++) {
      treeParams[i] = parseTree(treeValues.get(i));
    }

    int[] treeInfo = gbtreeModel.apply("tree_info").toIntArray();

    GBTree gbm = new GBTree();
    gbm.setNumClass(numClass);
    gbm.setNumFeature(numFeature);
    gbm.loadModel(treeParams, treeInfo);
    return gbm;
  }

  private static AbstractRegTree.Param parseTree(UValue tree) {
    UValue treeParam = tree.apply("tree_param");
    int numNodes = Integer.parseInt(paramString(treeParam, "num_nodes"));

    int sizeLeafVector = Integer.parseInt(paramString(treeParam, "size_leaf_vector"));
    if (sizeLeafVector > 1) {
      throw new IllegalArgumentException(
          "Vector-leaf / multi-target trees (size_leaf_vector = "
              + sizeLeafVector
              + ") are not supported.");
    }

    int[] leftChildren = tree.apply("left_children").toIntArray();
    int[] rightChildren = tree.apply("right_children").toIntArray();
    int[] splitIndices = tree.apply("split_indices").toIntArray();
    float[] splitConditions = tree.apply("split_conditions").toFloatArray();
    int[] defaultLeftRaw = tree.apply("default_left").toIntArray();
    boolean[] defaultLeft = new boolean[defaultLeftRaw.length];
    for (int i = 0; i < defaultLeftRaw.length; i++) {
      defaultLeft[i] = defaultLeftRaw[i] != 0;
    }
    float[] sumHessian = tree.apply("sum_hessian").toFloatArray();

    Categorical categorical = parseCategorical(tree, numNodes);

    return new AbstractRegTree.Param(
        numNodes,
        leftChildren,
        rightChildren,
        splitIndices,
        splitConditions,
        defaultLeft,
        sumHessian,
        categorical.isCategorical,
        categorical.categories);
  }

  private record Categorical(boolean[] isCategorical, int[][] categories) {}

  /**
   * Decodes categorical splits per io_utils.h / tree_model.cc. The flat {@code categories} array
   * holds explicit member category values; {@code categories_nodes} lists the categorical node ids
   * and {@code categories_segments}/{@code categories_sizes} give each node's offset+length into
   * {@code categories}. Returns (null, null) for purely numeric trees so numeric traversal stays on
   * the fast path.
   */
  private static Categorical parseCategorical(UValue tree, int numNodes) {
    Optional<UValue> catNodesValue = tree.get("categories_nodes");
    if (catNodesValue.isPresent()) {
      int[] catNodes = catNodesValue.get().toIntArray();
      if (catNodes.length > 0) {
        int[] categories = tree.apply("categories").toIntArray();
        int[] segments = tree.apply("categories_segments").toIntArray();
        int[] sizes = tree.apply("categories_sizes").toIntArray();

        boolean[] isCategorical = new boolean[numNodes];
        int[][] nodeCategories = new int[numNodes][];
        for (int k = 0; k < catNodes.length; k++) {
          int nodeId = catNodes[k];
          int begin = segments[k];
          int size = sizes[k];
          isCategorical[nodeId] = true;
          nodeCategories[nodeId] = Arrays.copyOfRange(categories, begin, begin + size);
        }
        return new Categorical(isCategorical, nodeCategories);
      }
    }
    return new Categorical(null, null);
  }

  /**
   * base_score is stored as a JSON string in probability space. XGBoost 3.1+ emits it as a
   * bracketed vector: a single element for single-output models (e.g. "[4.821127E-1]") and one
   * element per output group for multiclass (e.g. "[-5.275E-1,1.0009E0,-4.734E-1]"). Each element
   * is a per-group intercept, so all are parsed and returned in order.
   */
  private static float[] parseBaseScores(String raw) {
    String trimmed = raw.trim();
    if (trimmed.startsWith("[")) {
      trimmed = trimmed.substring(1);
      if (trimmed.endsWith("]")) {
        trimmed = trimmed.substring(0, trimmed.length() - 1);
      }
    }
    String[] parts = trimmed.split(",");
    float[] scores = new float[parts.length];
    for (int i = 0; i < parts.length; i++) {
      scores[i] = Float.parseFloat(parts[i].trim());
    }
    return scores;
  }

  /** learner_model_param and tree_param store all values as JSON strings. */
  static String paramString(UValue param, String key) {
    UValue value = param.apply(key);
    if (value instanceof UString s) {
      return s.value();
    }
    if (value instanceof UNumber n) {
      return Double.toString(n.value());
    }
    throw new IllegalArgumentException("Expected string param '" + key + "', got " + value);
  }
}
