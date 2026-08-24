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

import com.yelp.xgboost.parser.PredictorFactory;
import com.yelp.xgboost.testing.LEWriter;
import org.junit.Test;

/**
 * Pins the {@code num_pbuffer} prediction-buffer skip in {@link com.yelp.xgboost.gbm.GBTree}. That
 * branch is inert for every in-scope model (XGBoost 1.0+ always writes {@code num_pbuffer == 0}, so
 * the frozen golden fixtures never reach it), but the code remains as a safety net for pre-1.0
 * legacy bundles. This hand-packs a minimal legacy-binary model with {@code num_pbuffer != 0} and
 * {@code contain_extra_attrs != 0} (which the loader passes as {@code with_pbuffer}) so the skip
 * executes.
 *
 * <p>The model appends exactly {@code 2 * 4 * predBufferSize()} trailing bytes for the two buffer
 * blocks. {@link com.yelp.xgboost.parser.ModelReader#skip} throws when fewer bytes are available
 * than requested, so a correct load confirms {@code predBufferSize()} computed the expected size:
 * an off-by-one in that arithmetic would over- or under-run the trailing bytes and fail the load.
 */
public class GBTreePbufferSkipTest {

  private static final float LEAF_VALUE = 2.5f;
  private static final float BASE_SCORE = 0.0f;

  @Test
  public void skipsPredictionBufferOnLegacyBinaryWithPbuffer() {
    byte[] model = legacyBinaryWithPbuffer();

    var predictor = PredictorFactory.fromModelBytes(model);
    float prediction = predictor.predict(FVec.fromArray(new float[] {0.0f}))[0];

    assertEquals(
        "a single-leaf tree must predict base_score + leaf_value once the pbuffer is skipped",
        BASE_SCORE + LEAF_VALUE,
        prediction,
        0.0f);
  }

  /**
   * Serializes a one-tree, single-leaf gbtree in the pre-UBJSON binary layout the legacy {@link
   * com.yelp.xgboost.parser.ModelReader} path consumes: learner header, objective and booster
   * names, gbtree model param (with {@code num_pbuffer = 1}), the tree, tree_info, then the two
   * prediction buffer blocks the skip must consume.
   */
  private static byte[] legacyBinaryWithPbuffer() {
    LEWriter w = new LEWriter();

    w.bytes(new byte[] {0x62, 0x69, 0x6e, 0x66}); // "binf" magic, so base_score follows as a float
    w.f(BASE_SCORE);
    w.i(1); // num_feature
    w.i(0); // num_class (0 -> single output group)
    w.i(1); // contain_extra_attrs != 0 -> loadModel receives with_pbuffer = true
    w.i(0); // contain_eval_metrics
    w.i(1); // major_version (>= 1: base_score is mapped through probToMargin, identity here)
    w.i(0); // minor_version
    w.zeros(27 * 4); // reserved[27]

    w.str("reg:squarederror"); // objective: identity predTransform and probToMargin
    w.str("gbtree");

    w.i(1); // num_trees
    w.i(1); // num_roots
    w.i(1); // num_feature (deprecated)
    w.i(0); // padding
    w.l(1L); // num_pbuffer != 0
    w.i(1); // num_output_group (unused on read)
    w.i(0); // size_leaf_vector
    w.zeros(31 * 4); // reserved[31]
    w.i(0); // padding

    w.i(1); // tree num_roots
    w.i(1); // tree num_nodes
    w.i(0); // num_deleted
    w.i(0); // max_depth
    w.i(1); // num_feature
    w.i(0); // size_leaf_vector
    w.zeros(31 * 4); // reserved[31]
    w.i(-1); // node parent_
    w.i(-1); // node cleft_ == -1 marks a leaf
    w.i(-1); // node cright_
    w.i(0); // node sindex_
    w.f(LEAF_VALUE); // leaf value (read because cleft_ == -1)
    w.f(0.0f); // stats: loss_chg
    w.f(1.0f); // stats: sum_hess
    w.f(0.0f); // stats: base_weight
    w.i(0); // stats: leaf_child_cnt

    w.i(0); // tree_info[0] -> output group 0

    w.zeros(2 * 4 * predBufferSize()); // the two prediction-buffer blocks the skip consumes

    return w.toByteArray();
  }

  /**
   * Matches {@code GBTree.predBufferSize()}: {@code num_output_group * num_pbuffer *
   * (size_leaf_vector + 1)} for the header written above (1 * 1 * (0 + 1)).
   */
  private static int predBufferSize() {
    return 1 * 1 * (0 + 1);
  }
}
