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

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Feature vector scored by {@link Predictor}. Build one with the static factories: {@link
 * #fromArray} / {@link #fromArrayWithZeroAsMissing} for dense rows, {@link #fromMap} for sparse
 * rows.
 *
 * <p>NOTE(farrell|2019-10-23): This class was modified from the open source version it was forked
 * from to always store feature values as floats to resolve COREML-1051
 */
public interface FVec extends Serializable {
  /**
   * Gets index-th value.
   *
   * @param index index
   * @return value
   */
  Float fvalue(int index);

  /**
   * Builds an {@link FVec} from a dense vector where only NaN marks a missing feature, matching
   * xgboost's default {@code missing=NaN}.
   *
   * @param values float values
   * @return feature vector
   */
  static FVec fromArray(float[] values) {
    return new FVecFloatArrayImpl(values, false);
  }

  /**
   * Builds an {@link FVec} from a dense vector where only NaN marks a missing feature, matching
   * xgboost's default {@code missing=NaN}.
   *
   * @param values double values
   * @return feature vector
   */
  static FVec fromArray(double[] values) {
    return new FVecFloatArrayImpl(toFloatArray(values), false);
  }

  /**
   * Builds an {@link FVec} from a dense vector where both NaN and {@code 0.0} mark a missing
   * feature, for a model trained with {@code missing=0.0f}.
   *
   * @param values float values
   * @return feature vector
   */
  static FVec fromArrayWithZeroAsMissing(float[] values) {
    return new FVecFloatArrayImpl(values, true);
  }

  /**
   * Builds an {@link FVec} from a dense vector where both NaN and {@code 0.0} mark a missing
   * feature, for a model trained with {@code missing=0.0f}.
   *
   * @param values double values
   * @return feature vector
   */
  static FVec fromArrayWithZeroAsMissing(double[] values) {
    return new FVecFloatArrayImpl(toFloatArray(values), true);
  }

  /**
   * Builds an {@link FVec} from a map. Any absent index is treated as a missing feature.
   *
   * @param map map containing feature values
   * @return feature vector
   */
  static FVec fromMap(Map<Integer, ? extends Number> map) {
    Map<Integer, Float> floatMap = new HashMap<>();
    for (Map.Entry<Integer, ? extends Number> indexAndValue : map.entrySet()) {
      floatMap.put(indexAndValue.getKey(), indexAndValue.getValue().floatValue());
    }
    return new FVecMapImpl(floatMap);
  }

  /**
   * Builds an {@link FVec} from a sparse row whose absent indices hold 0.0, as when a sparse vector
   * is expanded to its dense form. Predictions match {@link #fromArray} over that dense row, or
   * {@link #fromArrayWithZeroAsMissing} when {@code treatsZeroAsNA} is set, while the memory and
   * build cost stay proportional to the number of entries rather than to {@code size}. Use {@link
   * #fromMap} instead when an absent index means missing.
   *
   * @param indices distinct feature indices of the stored entries
   * @param values values of the stored entries, aligned with {@code indices}
   * @param size number of features of the dense row
   * @param treatsZeroAsNA whether a 0.0 value, stored or absent, is missing
   * @return feature vector
   */
  static FVec fromSparse(int[] indices, float[] values, int size, boolean treatsZeroAsNA) {
    return new FVecSparseImpl(indices, values, size, treatsZeroAsNA);
  }

  /**
   * Double-valued variant of {@link #fromSparse(int[], float[], int, boolean)}.
   *
   * @param indices distinct feature indices of the stored entries
   * @param values values of the stored entries, aligned with {@code indices}
   * @param size number of features of the dense row
   * @param treatsZeroAsNA whether a 0.0 value, stored or absent, is missing
   * @return feature vector
   */
  static FVec fromSparse(int[] indices, double[] values, int size, boolean treatsZeroAsNA) {
    return new FVecSparseImpl(indices, toFloatArray(values), size, treatsZeroAsNA);
  }

  private static float[] toFloatArray(double[] values) {
    float[] float_values = new float[values.length];
    for (int i = 0; i < values.length; i++) {
      float_values[i] = (float) values[i];
    }
    return float_values;
  }

  class FVecMapImpl implements FVec {
    private final Map<Integer, ? extends Float> values;

    FVecMapImpl(Map<Integer, Float> values) {
      this.values = values;
    }

    @Override
    public Float fvalue(int index) {
      Float value = values.get(index);
      if (value == null || Float.isNaN(value)) {
        return null;
      }
      return value;
    }
  }

  class FVecFloatArrayImpl implements FVec {
    private final float[] values;
    private final boolean treatsZeroAsNA;

    FVecFloatArrayImpl(float[] values, boolean treatsZeroAsNA) {
      this.values = values;
      this.treatsZeroAsNA = treatsZeroAsNA;
    }

    @Override
    public Float fvalue(int index) {
      // Return null for any absent or missing value so tree traversal routes it the split's
      // default direction, matching native xgboost. A present value returned as-is would instead
      // fall through to a numeric comparison (NaN < split is false), ignoring the learned default
      // and silently going one way. Missing means: an out-of-range index; NaN, which is
      // xgboost's canonical missing marker (DMatrix default missing=NaN); or, when treatsZeroAsNA
      // is set (model trained with missing=0.0f), a 0.0 value.
      if (values.length <= index) {
        return null;
      }

      float result = values[index];
      if (Float.isNaN(result) || (treatsZeroAsNA && result == 0)) {
        return null;
      }

      return result;
    }
  }

  /**
   * Sparse row backed by an open-addressing table of primitive keys and values (linear probing,
   * Fibonacci hashing, load factor at most 0.5). An index absent from the table reads as 0.0, and
   * missing values follow {@link FVecFloatArrayImpl}.
   */
  class FVecSparseImpl implements FVec {
    private static final int EMPTY = -1;
    private static final int GOLDEN_RATIO = 0x9E3779B9;
    private final int[] keys;
    private final float[] slots;
    private final int mask;
    private final int shift;
    private final int size;
    private final boolean treatsZeroAsNA;

    FVecSparseImpl(int[] indices, float[] values, int size, boolean treatsZeroAsNA) {
      if (indices.length != values.length) {
        throw new IllegalArgumentException(
            "indices and values differ in length: " + indices.length + " vs " + values.length);
      }
      int capacity = Integer.highestOneBit(Math.max(indices.length, 1) * 2 - 1) << 1;
      this.keys = new int[capacity];
      this.slots = new float[capacity];
      this.mask = capacity - 1;
      this.shift = Integer.numberOfLeadingZeros(capacity) + 1;
      this.size = size;
      this.treatsZeroAsNA = treatsZeroAsNA;
      Arrays.fill(keys, EMPTY);
      for (int i = 0; i < indices.length; i++) {
        put(indices[i], values[i]);
      }
    }

    @Override
    public Float fvalue(int index) {
      if (size <= index) {
        return null;
      }
      int position = slot(index);
      float result = keys[position] == EMPTY ? 0.0f : slots[position];
      if (Float.isNaN(result) || (treatsZeroAsNA && result == 0)) {
        return null;
      }
      return result;
    }

    private void put(int index, float value) {
      int position = slot(index);
      keys[position] = index;
      slots[position] = value;
    }

    private int slot(int index) {
      int position = (index * GOLDEN_RATIO) >>> shift;
      while (keys[position] != index && keys[position] != EMPTY) {
        position = (position + 1) & mask;
      }
      return position;
    }
  }
}
