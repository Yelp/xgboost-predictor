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
package com.yelp.xgboost.gbm;

import com.yelp.xgboost.FVec;
import com.yelp.xgboost.parser.ModelReader;
import java.io.IOException;
import java.io.Serializable;

/** Linear booster implementation */
class GBLinear extends GBBase {

  private float[] weights;

  @Override
  public void loadModel(ModelReader reader, boolean ignored_with_pbuffer) throws IOException {
    new ModelParam(reader);
    long len = reader.readLong();
    if (len == 0) {
      weights = new float[(num_feature + 1) * num_output_group];
    } else {
      weights = reader.readFloatArray((int) len);
    }
  }

  @Override
  public float[] predict(FVec feat, int ntree_limit) {
    float[] preds = new float[num_output_group];
    for (int gid = 0; gid < num_output_group; ++gid) {
      preds[gid] = pred(feat, gid);
    }
    return preds;
  }

  @Override
  public float predictSingle(FVec feat, int ntree_limit) {
    if (num_output_group != 1) {
      throw new IllegalStateException(
          "Can't invoke predictSingle() because this model outputs multiple values: "
              + num_output_group);
    }
    return pred(feat, 0);
  }

  float pred(FVec feat, int gid) {
    float psum = bias(gid);
    Float featValue;
    for (int fid = 0; fid < num_feature; ++fid) {
      featValue = feat.fvalue(fid);
      if (featValue != null) {
        psum += featValue * weight(fid, gid);
      }
    }
    return psum;
  }

  @Override
  public int[] predictLeaf(FVec feat, int ntree_limit) {
    throw new UnsupportedOperationException("gblinear does not support predict leaf index");
  }

  float weight(int fid, int gid) {
    return weights[(fid * num_output_group) + gid];
  }

  float bias(int gid) {
    return weights[(num_feature * num_output_group) + gid];
  }

  static class ModelParam implements Serializable {
    /*! \brief reserved space */
    final int[] reserved;

    ModelParam(ModelReader reader) throws IOException {
      reader.readUnsignedInt(); // num_feature deprecated
      reader.readInt(); // num_output_group deprecated
      reserved = reader.readIntArray(32);
    }
  }
}
