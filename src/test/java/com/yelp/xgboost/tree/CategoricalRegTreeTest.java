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
package com.yelp.xgboost.tree;

import static org.junit.Assert.assertEquals;

import com.yelp.xgboost.FVec;
import org.junit.Test;

/**
 * Unit-pins the category-membership edge guards in {@link AbstractRegTree.Node}. XGBoost's {@code
 * common::Decision} (categorical.h) routes right only for a value that is a non-negative, integral,
 * in-range member of the split's category set; a negative, non-integral, or out-of-range ({@code >=
 * 16777216}) value routes left as one-hot encoding would. The parity tests feed only clean integer
 * categories {@code 0..49} plus NaN (which short-circuits earlier in {@link
 * AbstractRegTree.Node#next}), so these three guard branches are unreachable from them and are
 * pinned directly here.
 */
public class CategoricalRegTreeTest {

  private static final float LEFT = -1.0f;
  private static final float RIGHT = 1.0f;
  private static final float CATEGORICAL_CARDINALITY_LIMIT = 16777216f;

  /**
   * A single categorical split on feature 0 with category set {@code {1, 2}}. Members route right
   * (node 2), everything else routes left (node 1). {@code defaultLeft} is true so a missing value
   * would route left, but these cases all supply a present value to reach the membership test.
   */
  private static CategoricalRegTree oneSplitCategoricalTree() {
    int[] leftChildren = {1, -1, -1};
    int[] rightChildren = {2, -1, -1};
    int[] splitIndices = {0, 0, 0};
    float[] splitConditionsOrLeafValues = {0.0f, LEFT, RIGHT};
    boolean[] defaultLeft = {true, false, false};
    float[] sumHessian = {2.0f, 1.0f, 1.0f};
    boolean[] isCategorical = {true, false, false};
    int[][] categories = {{1, 2}, null, null};

    AbstractRegTree.Param param =
        new AbstractRegTree.Param(
            3,
            leftChildren,
            rightChildren,
            splitIndices,
            splitConditionsOrLeafValues,
            defaultLeft,
            sumHessian,
            isCategorical,
            categories);
    CategoricalRegTree tree = new CategoricalRegTree();
    tree.loadModel(param);
    return tree;
  }

  private static float leafValueFor(CategoricalRegTree tree, float feature) {
    return tree.getLeafValue(FVec.fromArray(new float[] {feature}));
  }

  @Test
  public void memberCategoryRoutesRight() {
    CategoricalRegTree tree = oneSplitCategoricalTree();
    assertEquals(
        "category 1 is in the set and must route right", RIGHT, leafValueFor(tree, 1.0f), 0.0f);
    assertEquals(
        "category 2 is in the set and must route right", RIGHT, leafValueFor(tree, 2.0f), 0.0f);
  }

  @Test
  public void nonMemberIntegerCategoryRoutesLeft() {
    CategoricalRegTree tree = oneSplitCategoricalTree();
    assertEquals(
        "category 0 is not in the set and must route left", LEFT, leafValueFor(tree, 0.0f), 0.0f);
    assertEquals(
        "category 5 is not in the set and must route left", LEFT, leafValueFor(tree, 5.0f), 0.0f);
  }

  @Test
  public void negativeCategoryRoutesLeft() {
    CategoricalRegTree tree = oneSplitCategoricalTree();
    assertEquals(
        "a negative value is never a category member and must route left",
        LEFT,
        leafValueFor(tree, -1.0f),
        0.0f);
  }

  @Test
  public void nonIntegralCategoryRoutesLeft() {
    CategoricalRegTree tree = oneSplitCategoricalTree();
    assertEquals(
        "a non-integral value is never a category member and must route left",
        LEFT,
        leafValueFor(tree, 1.5f),
        0.0f);
  }

  @Test
  public void outOfRangeCategoryRoutesLeft() {
    CategoricalRegTree tree = oneSplitCategoricalTree();
    assertEquals(
        "a value at the cardinality limit is out of range and must route left",
        LEFT,
        leafValueFor(tree, CATEGORICAL_CARDINALITY_LIMIT),
        0.0f);
    assertEquals(
        "a value past the cardinality limit is out of range and must route left",
        LEFT,
        leafValueFor(tree, CATEGORICAL_CARDINALITY_LIMIT + 100f),
        0.0f);
  }
}
