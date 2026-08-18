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
package com.yelp.xgboost;

import com.yelp.xgboost.gbm.GradBooster;
import com.yelp.xgboost.learner.ObjFunction;
import com.yelp.xgboost.parser.ModelReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;

/** Predicts using the Xgboost model. */
public class Predictor implements Serializable {
  private ModelParam mparam;
  private String name_obj;
  private String name_gbm;
  private ObjFunction obj;
  private GradBooster gbm;

  private float base_score;

  /**
   * Per-output-group intercept in margin space, or null when a single scalar {@link #base_score} is
   * shared across all groups. XGBoost 3.x stores one base_score per class for multiclass models.
   */
  private float[] base_margins;

  /** Raw serialized model bytes retained so a loaded model can be re-serialized losslessly. */
  private byte[] rawModel;

  /**
   * Instantiates with the Xgboost model
   *
   * @param in input stream
   * @throws IOException If an I/O error occurs
   */
  public Predictor(InputStream in) throws IOException {
    ModelReader reader = new ModelReader(in);

    mparam = new ModelParam(reader);
    name_obj = reader.readString();
    name_gbm = reader.readString();

    initObjGbm();

    gbm.loadModel(reader, mparam.contain_extra_attrs != 0);

    if (mparam.major_version >= 1) {
      base_score = obj.probToMargin(mparam.base_score);
    } else {
      base_score = mparam.base_score;
    }
  }

  /**
   * Instantiates from already-parsed model pieces (used by the JSON/UBJSON reader). Base margins
   * must be supplied in margin/link space; the caller (XGBoostModelParser) applies obj.probToMargin
   * to the probability-space values stored in learner_model_param before passing them here.
   *
   * <p>A single-element array is a scalar intercept shared across all output groups; a
   * multi-element array is one intercept per class (XGBoost 3.x multiclass), applied per group in
   * {@link #predictRaw(FVec, int)}.
   *
   * @param obj objective function
   * @param gbm gradient booster
   * @param baseMargins per-output-group global bias in margin space
   */
  public Predictor(ObjFunction obj, GradBooster gbm, float[] baseMargins) {
    this.obj = obj;
    this.gbm = gbm;
    this.base_score = baseMargins[0];
    this.base_margins = baseMargins.length > 1 ? baseMargins : null;
  }

  void initObjGbm() {
    obj = ObjFunction.fromName(name_obj);
    gbm = GradBooster.Factory.createGradBooster(name_gbm);
    gbm.setNumClass(mparam.num_class);
    gbm.setNumFeature(mparam.num_feature);
  }

  /**
   * Generates transformed predictions (probability space) for given feature vector, scoring all
   * trees. For margin-space output use {@link #predictRaw(FVec)}.
   *
   * @param feat feature vector
   * @return transformed prediction values
   */
  public float[] predict(FVec feat) {
    return predict(feat, 0);
  }

  /**
   * Generates transformed predictions (probability space) for given feature vector. For
   * margin-space output use {@link #predictRaw(FVec, int)}.
   *
   * @param feat feature vector
   * @param ntree_limit limit the number of trees used in prediction
   * @return transformed prediction values
   */
  public float[] predict(FVec feat, int ntree_limit) {
    return obj.predTransform(predictRaw(feat, base_score, ntree_limit));
  }

  float[] predictRaw(FVec feat, float base_score, int ntree_limit) {
    float[] preds = gbm.predict(feat, ntree_limit);
    for (int i = 0; i < preds.length; i++) {
      preds[i] += baseMarginFor(i, base_score);
    }
    return preds;
  }

  /**
   * The intercept for output group {@code group}: the per-group value when the model carries one
   * base_score per class, otherwise the shared scalar.
   */
  private float baseMarginFor(int group, float base_score) {
    if (base_margins != null && group < base_margins.length) {
      return base_margins[group];
    }
    return base_score;
  }

  /**
   * Generates raw margin-space predictions for given feature vector, scoring all trees. This is the
   * untransformed counterpart of {@link #predict(FVec)}.
   *
   * @param feat feature vector
   * @return raw margin-space prediction values
   */
  public float[] predictRaw(FVec feat) {
    return predictRaw(feat, 0);
  }

  /**
   * Generates raw margin-space predictions for given feature vector. This is the untransformed
   * counterpart of {@link #predict(FVec, int)}.
   *
   * @param feat feature vector
   * @param ntree_limit limit the number of trees used in prediction
   * @return raw margin-space prediction values
   */
  public float[] predictRaw(FVec feat, int ntree_limit) {
    return predictRaw(feat, base_score, ntree_limit);
  }

  /**
   * Convert margin value to transformed prediction.
   *
   * @param rawPreds raw prediction (margin-space) values
   * @return prediction values
   */
  float[] predTransform(float[] rawPreds) {
    return obj.predTransform(rawPreds);
  }

  /**
   * Generates a transformed single-value prediction (probability space) for given feature vector,
   * scoring all trees. For margin-space output use {@link #predictSingleRaw(FVec)}.
   *
   * <p>This method only works when the model outputs single value.
   *
   * @param feat feature vector
   * @return transformed prediction value
   */
  public float predictSingle(FVec feat) {
    return predictSingle(feat, 0);
  }

  /**
   * Generates a transformed single-value prediction (probability space) for given feature vector.
   * For margin-space output use {@link #predictSingleRaw(FVec, int)}.
   *
   * <p>This method only works when the model outputs single value.
   *
   * @param feat feature vector
   * @param ntree_limit limit the number of trees used in prediction
   * @return transformed prediction value
   */
  public float predictSingle(FVec feat, int ntree_limit) {
    return obj.predTransform(predictSingleRaw(feat, ntree_limit));
  }

  /**
   * Generates a raw margin-space single-value prediction for given feature vector, scoring all
   * trees. This is the untransformed counterpart of {@link #predictSingle(FVec)}.
   *
   * <p>This method only works when the model outputs single value.
   *
   * @param feat feature vector
   * @return raw margin-space prediction value
   */
  public float predictSingleRaw(FVec feat) {
    return predictSingleRaw(feat, 0);
  }

  /**
   * Generates a raw margin-space single-value prediction for given feature vector. This is the
   * untransformed counterpart of {@link #predictSingle(FVec, int)}.
   *
   * <p>This method only works when the model outputs single value.
   *
   * @param feat feature vector
   * @param ntree_limit limit the number of trees used in prediction
   * @return raw margin-space prediction value
   */
  public float predictSingleRaw(FVec feat, int ntree_limit) {
    return gbm.predictSingle(feat, ntree_limit) + base_score;
  }

  /**
   * Predicts leaf index of each tree.
   *
   * @param feat feature vector
   * @return leaf indexes
   */
  public int[] predictLeaf(FVec feat) {
    return predictLeaf(feat, 0);
  }

  /**
   * Predicts leaf index of each tree.
   *
   * @param feat feature vector
   * @param ntree_limit limit
   * @return leaf indexes
   */
  public int[] predictLeaf(FVec feat, int ntree_limit) {
    return gbm.predictLeaf(feat, ntree_limit);
  }

  /** Parameters. */
  static class ModelParam implements Serializable {
    /* \brief global bias */
    final float base_score;
    /* \brief number of features  */
    final /* unsigned */ int num_feature;
    /* \brief number of class, if it is multi-class classification  */
    final int num_class;
    /*!
     * \brief whether the model carries an appended attribute map. Historically this offset held a
     * pbuffer flag, hence its use below as loadModel's with_pbuffer argument; that path is inert
     * for XGBoost 1.0+ because num_pbuffer is always 0, so the value never changes prediction.
     */
    final int contain_extra_attrs;
    /*! \brief Model contain eval metrics */
    private final int contain_eval_metrics;
    /*! \brief the version of XGBoost. */
    private final int major_version;
    private final int minor_version;
    /*! \brief reserved field */
    final int[] reserved;

    ModelParam(ModelReader reader) throws IOException {
      byte[] first4Bytes = reader.readByteArray(4);
      if (first4Bytes[0] == 0x62
          && first4Bytes[1] == 0x69
          && first4Bytes[2] == 0x6e
          && first4Bytes[3] == 0x66) {
        // The "binf" signature (62 69 6e 66) was added in XGBoost 1.3.0. Models from
        // 1.0.x through 1.2.x omit it and store base_score as the leading 4 bytes.
        base_score = reader.readFloat();
      } else {
        base_score = reader.asFloat(first4Bytes);
      }
      num_feature = reader.readUnsignedInt();
      num_class = reader.readInt();
      contain_extra_attrs = reader.readInt();
      this.contain_eval_metrics = reader.readInt();
      this.major_version = reader.readUnsignedInt();
      this.minor_version = reader.readUnsignedInt();
      this.reserved = reader.readIntArray(27);
    }
  }

  public float getBaseScore() {
    return base_score;
  }

  /** Sets the raw serialized model bytes (see {@link #getRawModel()}). */
  public void setRawModel(byte[] rawModel) {
    this.rawModel = rawModel;
  }

  /**
   * The raw serialized model bytes this predictor was loaded from, or null if unavailable. Used to
   * re-serialize a loaded model losslessly (the parsed tree structures are not written back).
   */
  public byte[] getRawModel() {
    return rawModel;
  }
}
