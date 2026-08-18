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
import static org.junit.Assert.assertNotEquals;

import com.yelp.xgboost.learner.ObjFunction;
import com.yelp.xgboost.parser.PredictorFactory;
import com.yelp.xgboost.testing.LEWriter;
import org.junit.Test;

/**
 * Pins two legacy-binary branches no frozen golden fixture reaches, because every captured golden
 * is an XGBoost 1.x/2.x gbtree with non-zero-length weights:
 *
 * <ul>
 *   <li>{@code Predictor.ModelParam} with {@code major_version < 1}: pre-1.0 models store
 *       base_score already in margin space, so the loader must use it verbatim rather than applying
 *       {@code obj.probToMargin}.
 *   <li>{@code GBLinear.loadModel} with a zero-length weight block: a degenerate linear model whose
 *       weights default to an all-zero vector, so every prediction collapses to base_score.
 * </ul>
 */
public class LegacyBinaryEdgeCasesTest {

  private static final float LEAF_VALUE = 2.5f;

  /**
   * base_score under reg:logistic, whose probToMargin is a non-identity logit. A major_version-0
   * model must expose this value unchanged; a &gt;=1 model would map it through the logit.
   */
  private static final float PROB_SPACE_BASE_SCORE = 0.6f;

  @Test
  public void preOneMajorVersionUsesBaseScoreVerbatim() {
    var predictor = PredictorFactory.fromModelBytes(legacyGbtreeMajorVersionZero());

    assertEquals(
        "major_version < 1 must expose base_score verbatim (already in margin space)",
        PROB_SPACE_BASE_SCORE,
        predictor.getBaseScore(),
        0.0f);
    assertNotEquals(
        "the verbatim path must differ from the probToMargin the >=1 path would apply",
        ObjFunction.fromName("reg:logistic").probToMargin(PROB_SPACE_BASE_SCORE),
        predictor.getBaseScore(),
        1e-4f);

    float prediction = predictor.predictRaw(FVec.fromArray(new float[] {0.0f}))[0];
    assertEquals(
        "margin must be base_score + leaf_value with the verbatim intercept",
        PROB_SPACE_BASE_SCORE + LEAF_VALUE,
        prediction,
        1e-5f);
  }

  @Test
  public void gblinearZeroLengthWeightsDefaultToAllZero() {
    var predictor = PredictorFactory.fromModelBytes(legacyGblinearZeroWeights());

    float first = predictor.predict(row(0.3f, 0.9f))[0];
    float second = predictor.predict(row(-4.0f, 100.0f))[0];

    assertEquals(
        "all-zero weights make every feature contribution vanish, leaving base_score",
        0.0f,
        first,
        0.0f);
    assertEquals(
        "an all-zero-weight linear model is constant regardless of the input row",
        first,
        second,
        0.0f);
  }

  private static FVec row(float... values) {
    return FVec.fromArray(values);
  }

  /**
   * A one-tree, single-leaf gbtree with no {@code "binf"} magic and {@code major_version = 0}, so
   * base_score is the leading little-endian float and the loader takes the verbatim branch.
   */
  private static byte[] legacyGbtreeMajorVersionZero() {
    LEWriter w = new LEWriter();

    w.f(PROB_SPACE_BASE_SCORE); // no "binf" magic: leading float is base_score
    w.i(1); // num_feature
    w.i(0); // num_class
    w.i(0); // contain_extra_attrs
    w.i(0); // contain_eval_metrics
    w.i(0); // major_version == 0 -> base_score used verbatim
    w.i(0); // minor_version
    w.zeros(27 * 4); // reserved[27]

    w.str("reg:logistic"); // non-identity probToMargin, so the branch choice is observable
    w.str("gbtree");

    w.i(1); // num_trees
    w.i(1); // num_roots
    w.i(1); // num_feature (deprecated)
    w.i(0); // padding
    w.l(0L); // num_pbuffer == 0 -> no skip
    w.i(1); // num_output_group (unused on read)
    w.i(0); // size_leaf_vector
    w.zeros(31 * 4); // reserved[31]
    w.i(0); // padding

    appendSingleLeafTree(w);

    w.i(0); // tree_info[0] -> output group 0
    return w.toByteArray();
  }

  private static void appendSingleLeafTree(LEWriter w) {
    w.i(1); // tree num_roots
    w.i(1); // tree num_nodes
    w.i(0); // num_deleted
    w.i(0); // max_depth
    w.i(1); // num_feature
    w.i(0); // size_leaf_vector
    w.zeros(31 * 4); // reserved[31]
    w.i(-1); // parent_
    w.i(-1); // cleft_ == -1 marks a leaf
    w.i(-1); // cright_
    w.i(0); // sindex_
    w.f(LEAF_VALUE); // leaf value
    w.f(0.0f); // stats: loss_chg
    w.f(1.0f); // stats: sum_hess
    w.f(0.0f); // stats: base_weight
    w.i(0); // stats: leaf_child_cnt
  }

  /**
   * A legacy gblinear model whose weight block declares length 0, forcing {@code
   * GBLinear.loadModel} to allocate an all-zero weight/bias vector. Uses reg:squarederror (identity
   * transform) with base_score 0 so the resulting prediction is exactly the zero intercept.
   */
  private static byte[] legacyGblinearZeroWeights() {
    LEWriter w = new LEWriter();

    w.bytes(new byte[] {0x62, 0x69, 0x6e, 0x66}); // "binf" magic
    w.f(0.0f); // base_score
    w.i(2); // num_feature
    w.i(0); // num_class -> num_output_group 1
    w.i(0); // contain_extra_attrs
    w.i(0); // contain_eval_metrics
    w.i(1); // major_version
    w.i(0); // minor_version
    w.zeros(27 * 4); // reserved[27]

    w.str("reg:squarederror"); // identity predTransform and probToMargin
    w.str("gblinear");

    w.i(2); // GBLinear.ModelParam num_feature (deprecated)
    w.i(1); // num_output_group (deprecated)
    w.zeros(32 * 4); // reserved[32]

    w.l(0L); // weight-block length 0 -> all-zero weights fallback
    return w.toByteArray();
  }
}
