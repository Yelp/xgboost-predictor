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
package com.yelp.xgboost.testing;

import com.yelp.xgboost.Predictor;
import com.yelp.xgboost.parser.PredictorFactory;
import java.io.ByteArrayOutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;

/**
 * Trains xgboost4j Boosters on small synthetic datasets and loads them through the same {@link
 * PredictorFactory} the runtime uses. Parity tests compare the pure-JVM {@link Predictor} against
 * the native Booster on identical inputs, without any Spark or external-fixture dependency.
 */
public final class BoosterTestUtils {

  public static final int NUM_FEATURES = 12;
  public static final int NUM_ROWS = 120;
  public static final int CATEGORICAL_NUM_FEATURES = 3;

  private BoosterTestUtils() {}

  private static Map<String, Object> commonParams() {
    Map<String, Object> params = new HashMap<>();
    params.put("eta", 0.3);
    params.put("max_depth", 2);
    return params;
  }

  /** A separable binary dataset: label 1 when the summed features exceed half the feature count. */
  public static DMatrix binomialDataset() throws Exception {
    Random random = new Random(1L);
    float[] data = new float[NUM_ROWS * NUM_FEATURES];
    float[] labels = new float[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      float sum = 0f;
      for (int j = 0; j < NUM_FEATURES; j++) {
        float v = random.nextFloat();
        data[i * NUM_FEATURES + j] = v;
        sum += v;
      }
      labels[i] = sum > NUM_FEATURES / 2.0f ? 1.0f : 0.0f;
    }
    DMatrix dmatrix = new DMatrix(data, NUM_ROWS, NUM_FEATURES, Float.NaN);
    dmatrix.setLabel(labels);
    return dmatrix;
  }

  /** A three-class dataset: class chosen by which third of the summed features the row lands in. */
  public static DMatrix multinomialDataset() throws Exception {
    Random random = new Random(2L);
    float[] data = new float[NUM_ROWS * NUM_FEATURES];
    float[] labels = new float[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      float sum = 0f;
      for (int j = 0; j < NUM_FEATURES; j++) {
        float v = random.nextFloat();
        data[i * NUM_FEATURES + j] = v;
        sum += v;
      }
      float mean = sum / NUM_FEATURES;
      labels[i] = mean < 0.4f ? 0.0f : (mean < 0.6f ? 1.0f : 2.0f);
    }
    DMatrix dmatrix = new DMatrix(data, NUM_ROWS, NUM_FEATURES, Float.NaN);
    dmatrix.setLabel(labels);
    return dmatrix;
  }

  public static Booster trainBinaryBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = commonParams();
    params.put("objective", "binary:logistic");
    return XGBoost.train(dataset, params, 15, new HashMap<>(), null, null);
  }

  public static Booster trainMultinomialBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = commonParams();
    params.put("objective", "multi:softprob");
    params.put("num_class", 3);
    return XGBoost.train(dataset, params, 15, new HashMap<>(), null, null);
  }

  public static Booster trainMultinomialSoftmaxBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = commonParams();
    params.put("objective", "multi:softmax");
    params.put("num_class", 3);
    return XGBoost.train(dataset, params, 15, new HashMap<>(), null, null);
  }

  public static Booster trainRegressionBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = commonParams();
    params.put("objective", "reg:squarederror");
    return XGBoost.train(dataset, params, 15, new HashMap<>(), null, null);
  }

  /**
   * A binary dataset with roughly a third of its cells set to NaN (xgboost's default missing
   * marker) so training learns a non-trivial default direction per split rather than defaulting
   * every node the same way. This is what makes {@link #trainSparseBinaryBooster} produce trees
   * whose missing routing actually diverges from a naive {@code value < split} comparison, which is
   * the behavior the NaN parity test must pin.
   */
  public static float[] sparseData() {
    Random random = new Random(101L);
    float[] data = new float[NUM_ROWS * NUM_FEATURES];
    for (int i = 0; i < NUM_ROWS; i++) {
      for (int j = 0; j < NUM_FEATURES; j++) {
        data[i * NUM_FEATURES + j] = random.nextFloat() < 0.33f ? Float.NaN : random.nextFloat();
      }
    }
    return data;
  }

  public static DMatrix sparseBinomialDataset() throws Exception {
    float[] data = sparseData();
    float[] labels = new float[NUM_ROWS];
    for (int i = 0; i < NUM_ROWS; i++) {
      float sum = 0f;
      int present = 0;
      for (int j = 0; j < NUM_FEATURES; j++) {
        float v = data[i * NUM_FEATURES + j];
        if (!Float.isNaN(v)) {
          sum += v;
          present++;
        }
      }
      labels[i] = (present > 0 && sum / present > 0.5f) ? 1.0f : 0.0f;
    }
    DMatrix dmatrix = new DMatrix(data, NUM_ROWS, NUM_FEATURES, Float.NaN);
    dmatrix.setLabel(labels);
    return dmatrix;
  }

  /**
   * Trains deeper trees ({@code max_depth} 5) so many internal splits are reached, maximizing the
   * chance a present NaN hits a split whose learned default direction contradicts a plain numeric
   * comparison.
   */
  public static Booster trainSparseBinaryBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = new HashMap<>();
    params.put("objective", "binary:logistic");
    params.put("eta", 0.3);
    params.put("max_depth", 5);
    return XGBoost.train(dataset, params, 25, new HashMap<>(), null, null);
  }

  /**
   * Trains a single-output booster on the binomial dataset under an arbitrary objective, so
   * objectives that share the identity/exp prediction transform can be pinned end-to-end against
   * native without a bespoke dataset each. count:poisson needs non-negative labels (the 0/1
   * binomial labels satisfy this).
   */
  public static Booster trainBoosterWithObjective(DMatrix dataset, String objective)
      throws Exception {
    Map<String, Object> params = commonParams();
    params.put("objective", objective);
    return XGBoost.train(dataset, params, 15, new HashMap<>(), null, null);
  }

  /** reg:gamma requires strictly-positive labels, so labels are shifted by +1 before training. */
  public static Booster trainGammaBooster(DMatrix dataset) throws Exception {
    int rows = (int) dataset.rowNum();
    int[] allRows = new int[rows];
    for (int i = 0; i < rows; i++) {
      allRows[i] = i;
    }
    DMatrix shifted = dataset.slice(allRows);
    float[] labels = dataset.getLabel();
    for (int i = 0; i < labels.length; i++) {
      labels[i] += 1.0f;
    }
    shifted.setLabel(labels);
    Map<String, Object> params = commonParams();
    params.put("objective", "reg:gamma");
    return XGBoost.train(shifted, params, 15, new HashMap<>(), null, null);
  }

  /**
   * A tiny synthetic dataset whose third feature is categorical with cardinality 4.
   * tree_method=hist plus max_cat_to_onehot &lt; cardinality forces partition-based categorical
   * splits (split_type=1 + categories arrays in the saved model), exercising the categorical
   * traversal path.
   */
  public static DMatrix categoricalDataset() throws Exception {
    int rows = 40;
    float[] data = new float[rows * CATEGORICAL_NUM_FEATURES];
    float[] labels = new float[rows];
    for (int i = 0; i < rows; i++) {
      int cat = i % 4;
      data[i * 3] = (i % 7) * 0.1f;
      data[i * 3 + 1] = (i % 5) * 0.2f;
      data[i * 3 + 2] = cat;
      labels[i] = (cat == 1 || cat == 2) ? 1.0f : 0.0f;
    }
    DMatrix dmatrix = new DMatrix(data, rows, CATEGORICAL_NUM_FEATURES, Float.NaN);
    dmatrix.setLabel(labels);
    dmatrix.setFeatureTypes(new String[] {"q", "q", "c"});
    return dmatrix;
  }

  public static Booster trainCategoricalBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = new HashMap<>();
    params.put("objective", "binary:logistic");
    params.put("tree_method", "hist");
    params.put("max_cat_to_onehot", 2);
    params.put("max_depth", 4);
    params.put("eta", 0.3);
    return XGBoost.train(dataset, params, 15, new HashMap<>(), null, null);
  }

  /**
   * A deep ({@code max_depth} 8), many-tree (200 rounds) binary booster. Purely numeric trees are
   * repacked into {@link com.yelp.xgboost.tree.PreorderRegTree}'s primitive int array, so large
   * deep trees are what exercise that repacking at scale where index arithmetic bugs would surface.
   */
  public static Booster trainDeepBinaryBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = new HashMap<>();
    params.put("objective", "binary:logistic");
    params.put("eta", 0.1);
    params.put("max_depth", 8);
    return XGBoost.train(dataset, params, 200, new HashMap<>(), null, null);
  }

  public static final int HIGH_CARD_CATEGORIES = 50;
  public static final int HIGH_CARD_NUM_FEATURES = 2;

  /**
   * A categorical dataset whose single feature has cardinality 50, spanning past the 32-category
   * bitfield word boundary that xgboost uses to store category sets. The label is 1 for an
   * arbitrary scattered subset of categories (including some above index 32) so the learned split
   * sets are non-trivial in both bitfield words.
   */
  public static DMatrix highCardinalityCategoricalDataset() throws Exception {
    int rowsPerCategory = 8;
    int rows = HIGH_CARD_CATEGORIES * rowsPerCategory;
    float[] data = new float[rows * HIGH_CARD_NUM_FEATURES];
    float[] labels = new float[rows];
    for (int i = 0; i < rows; i++) {
      int cat = i % HIGH_CARD_CATEGORIES;
      data[i * HIGH_CARD_NUM_FEATURES] = (i % 11) * 0.05f;
      data[i * HIGH_CARD_NUM_FEATURES + 1] = cat;
      labels[i] = isPositiveCategory(cat) ? 1.0f : 0.0f;
    }
    DMatrix dmatrix = new DMatrix(data, rows, HIGH_CARD_NUM_FEATURES, Float.NaN);
    dmatrix.setLabel(labels);
    dmatrix.setFeatureTypes(new String[] {"q", "c"});
    return dmatrix;
  }

  /** A scattered positive-category set straddling both 32-bit words of the category bitfield. */
  public static boolean isPositiveCategory(int cat) {
    return cat % 3 == 0 || cat == 33 || cat == 40 || cat == 49;
  }

  public static Booster trainHighCardinalityCategoricalBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = new HashMap<>();
    params.put("objective", "binary:logistic");
    params.put("tree_method", "hist");
    params.put("max_cat_to_onehot", 1);
    params.put("max_depth", 6);
    params.put("eta", 0.3);
    return XGBoost.train(dataset, params, 30, new HashMap<>(), null, null);
  }

  /** A single boosting round, so the model carries exactly one tree per output group. */
  public static Booster trainSingleTreeBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = commonParams();
    params.put("objective", "binary:logistic");
    return XGBoost.train(dataset, params, 1, new HashMap<>(), null, null);
  }

  /**
   * A single boosting round with a prohibitive {@code gamma} (min split loss), so no split clears
   * the gain threshold and each tree collapses to a root-only leaf (a stump). Exercises the
   * degenerate tree where the traversal must return the root leaf without ever taking a branch.
   */
  public static Booster trainStumpBooster(DMatrix dataset) throws Exception {
    Map<String, Object> params = new HashMap<>();
    params.put("objective", "binary:logistic");
    params.put("eta", 0.3);
    params.put("max_depth", 6);
    params.put("gamma", 1.0e9);
    return XGBoost.train(dataset, params, 1, new HashMap<>(), null, null);
  }

  public static Predictor predictorFromBooster(Booster booster) throws Exception {
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    booster.saveModel(out);
    return PredictorFactory.fromModelBytes(out.toByteArray());
  }
}
