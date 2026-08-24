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
package com.yelp.xgboost.learner;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.junit.Test;

/**
 * Registry coverage for objectives a legacy 1.x binary could carry. {@code rank:map} joins the
 * pairwise/ndcg ranking objectives as an identity transform, and {@code binary:hinge} thresholds
 * the margin at zero. Both were absent from the registry and would have thrown on load.
 */
public class ObjFunctionTest {

  private static final float EPS = 1e-6f;

  @Test
  public void rankMapIsRegisteredAsAnIdentityTransform() {
    ObjFunction obj = ObjFunction.fromName("rank:map");
    assertEquals(0.75f, obj.predTransform(0.75f), 0.0f);
    assertArrayEquals(
        new float[] {-2.0f, 3.5f}, obj.predTransform(new float[] {-2.0f, 3.5f}), 0.0f);
    assertEquals(0.4f, obj.probToMargin(0.4f), 0.0f);
  }

  @Test
  public void binaryHingeThresholdsTheMarginAtZero() {
    ObjFunction obj = ObjFunction.fromName("binary:hinge");
    assertEquals(1.0f, obj.predTransform(0.01f), 0.0f);
    assertEquals(0.0f, obj.predTransform(0.0f), 0.0f);
    assertEquals(0.0f, obj.predTransform(-3.0f), 0.0f);
    assertArrayEquals(
        new float[] {1.0f, 0.0f, 0.0f, 1.0f},
        obj.predTransform(new float[] {2.5f, -0.5f, 0.0f, 0.1f}),
        0.0f);
  }

  @Test
  public void logisticProbToMarginInvertsSigmoid() {
    ObjFunction obj = ObjFunction.fromName("binary:logistic");
    float margin = obj.probToMargin(0.7f);
    assertEquals(0.7f, obj.predTransform(margin), EPS);
  }

  /**
   * The log-link objectives store base_score in probability space and map it to margin space via
   * {@code log} in the Predictor constructor. A wrong link would silently shift every prediction,
   * so pin that {@code predTransform(probToMargin(p)) == p} for the whole exp family.
   */
  @Test
  public void expFamilyProbToMarginInvertsExp() {
    for (String name : new String[] {"reg:gamma", "reg:tweedie", "count:poisson"}) {
      ObjFunction obj = ObjFunction.fromName(name);
      for (float prob : new float[] {0.05f, 0.5f, 1.0f, 2.5f, 10.0f}) {
        float margin = obj.probToMargin(prob);
        assertEquals(
            name + " probToMargin must invert predTransform at " + prob,
            prob,
            obj.predTransform(margin),
            EPS * Math.max(1.0f, prob));
      }
    }
  }

  /**
   * Multiclass objectives are inherently vector-valued (one score per class), so the scalar {@code
   * predTransform(float)} has no meaning and must reject rather than return a bogus value. A caller
   * reaching it signals a wrong prediction path, which this pins for both softmax and softprob.
   */
  @Test
  public void multiclassScalarPredTransformIsUnsupported() {
    for (String name : new String[] {"multi:softmax", "multi:softprob"}) {
      ObjFunction obj = ObjFunction.fromName(name);
      assertThrows(
          name + " scalar predTransform must be unsupported",
          UnsupportedOperationException.class,
          () -> obj.predTransform(0.5f));
    }
  }

  @Test
  public void unknownObjectiveIsRejected() {
    assertThrows(IllegalArgumentException.class, () -> ObjFunction.fromName("reg:nonexistent"));
  }
}
