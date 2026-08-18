/*
 * Copyright 2015-2016 Committers of xgboost-predictor-java (https://github.com/komiya-atsushi/xgboost-predictor-java)
 *
 * Copyright 2026 Yelp Inc.
 *
 * Vendored and extended for XGBoost 3.x JSON/UBJSON models.
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

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/** Objective function implementations. */
public class ObjFunction implements Serializable {
  private static final Map<String, ObjFunction> FUNCTIONS = new HashMap<>();

  static {
    register("rank:pairwise", new ObjFunction());
    register("rank:ndcg", new ObjFunction());
    register("rank:map", new ObjFunction());
    register("binary:logistic", new RegLossObjLogistic());
    register("reg:logistic", new RegLossObjLogistic());
    register("binary:logitraw", new ObjFunction());
    register("binary:hinge", new HingeObj());
    register("multi:softmax", new SoftmaxMultiClassObjClassify());
    register("multi:softprob", new SoftmaxMultiClassObjProb());
    register("reg:linear", new ObjFunction());
    register("reg:squarederror", new ObjFunction());
    // Identity prediction link (the log is in the loss/gradient, not the predict transform).
    register("reg:squaredlogerror", new ObjFunction());
    register("count:poisson", new RegLossObjExpFamily());
    register("reg:tweedie", new RegLossObjExpFamily());
    register("reg:gamma", new RegLossObjExpFamily());
  }

  /**
   * Gets {@link ObjFunction} from given name.
   *
   * @param name name of objective function
   * @return objective function
   */
  public static ObjFunction fromName(String name) {
    ObjFunction result = FUNCTIONS.get(name);
    if (result == null) {
      throw new IllegalArgumentException(name + " is not supported objective function.");
    }
    return result;
  }

  private static void register(String name, ObjFunction objFunction) {
    FUNCTIONS.put(name, objFunction);
  }

  /**
   * Transforms prediction values.
   *
   * @param preds prediction
   * @return transformed values
   */
  public float[] predTransform(float[] preds) {
    // do nothing
    return preds;
  }

  /**
   * Transforms a prediction value.
   *
   * @param pred prediction
   * @return transformed value
   */
  public float predTransform(float pred) {
    // do nothing
    return pred;
  }

  public float probToMargin(float prob) {
    // do nothing
    return prob;
  }

  /** Objective functions that need exp transformation. E.g., poisson, gamma, tweedie */
  static class RegLossObjExpFamily extends ObjFunction {
    @Override
    public float[] predTransform(float[] preds) {
      for (int i = 0; i < preds.length; i++) {
        preds[i] = exp(preds[i]);
      }
      return preds;
    }

    /** Log link: base_score is stored in probability space and mapped to margin space via log. */
    @Override
    public float probToMargin(float prob) {
      return (float) Math.log(prob);
    }

    @Override
    public float predTransform(float pred) {
      return exp(pred);
    }

    float exp(float x) {
      return (float) Math.exp(x);
    }
  }

  /** Hinge loss: the margin is thresholded at zero to a hard 0/1 label. */
  static class HingeObj extends ObjFunction {
    @Override
    public float[] predTransform(float[] preds) {
      for (int i = 0; i < preds.length; i++) {
        preds[i] = predTransform(preds[i]);
      }
      return preds;
    }

    @Override
    public float predTransform(float pred) {
      return pred > 0.0f ? 1.0f : 0.0f;
    }
  }

  /** Logistic regression. */
  static class RegLossObjLogistic extends ObjFunction {
    @Override
    public float[] predTransform(float[] preds) {
      for (int i = 0; i < preds.length; i++) {
        preds[i] = sigmoid(preds[i]);
      }
      return preds;
    }

    @Override
    public float predTransform(float pred) {
      return sigmoid(pred);
    }

    float sigmoid(float x) {
      return (1.0f / (1.0f + (float) Math.exp(-(x))));
    }

    @Override
    public float probToMargin(float prob) {
      return (float) -Math.log(1.0f / prob - 1.0f);
    }
  }

  /** Multiclass classification. */
  static class SoftmaxMultiClassObjClassify extends ObjFunction {
    @Override
    public float[] predTransform(float[] preds) {
      int maxIndex = 0;
      float max = preds[0];
      for (int i = 1; i < preds.length; i++) {
        if (max < preds[i]) {
          maxIndex = i;
          max = preds[i];
        }
      }

      return new float[] {maxIndex};
    }

    @Override
    public float predTransform(float pred) {
      throw new UnsupportedOperationException();
    }
  }

  /** Multiclass classification (predicted probability). */
  static class SoftmaxMultiClassObjProb extends ObjFunction {
    @Override
    public float[] predTransform(float[] preds) {
      float max = preds[0];
      for (int i = 1; i < preds.length; i++) {
        max = Math.max(preds[i], max);
      }

      float sum = 0;
      for (int i = 0; i < preds.length; i++) {
        preds[i] = exp(preds[i] - max);
        sum += preds[i];
      }

      for (int i = 0; i < preds.length; i++) {
        preds[i] /= (float) sum;
      }

      return preds;
    }

    @Override
    public float predTransform(float pred) {
      throw new UnsupportedOperationException();
    }

    float exp(float x) {
      return (float) Math.exp(x);
    }
  }
}
