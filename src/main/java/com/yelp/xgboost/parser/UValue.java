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

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Generic value tree produced by {@link UBJSONReader} and consumed by {@link XGBoostModelParser}.
 *
 * <p>XGBoost's JSON and UBJSON encodings share an identical key/structure schema, so the parser
 * works against this format-agnostic tree regardless of which decoder produced it.
 */
sealed interface UValue
    permits UValue.UObject,
        UValue.UArray,
        UValue.UString,
        UValue.UNumber,
        UValue.UBool,
        UValue.UNull,
        UValue.UIntArray,
        UValue.UFloatArray {

  default Map<String, UValue> obj() {
    return ((UObject) this).fields();
  }

  default List<UValue> arr() {
    return ((UArray) this).elements();
  }

  default String str() {
    return ((UString) this).value();
  }

  default UValue apply(String key) {
    UValue value = obj().get(key);
    if (value == null) {
      throw new java.util.NoSuchElementException("key not found: " + key);
    }
    return value;
  }

  default Optional<UValue> get(String key) {
    if (this instanceof UObject o) {
      return Optional.ofNullable(o.fields().get(key));
    }
    return Optional.empty();
  }

  default int asInt() {
    return (int) ((UNumber) this).value();
  }

  default long asLong() {
    return (long) ((UNumber) this).value();
  }

  default double asDouble() {
    return ((UNumber) this).value();
  }

  /** Homogeneous numeric arrays decode straight to primitive arrays for the tree parser. */
  default int[] toIntArray() {
    if (this instanceof UIntArray a) {
      return a.values();
    }
    if (this instanceof UArray a) {
      int[] out = new int[a.elements().size()];
      for (int i = 0; i < out.length; i++) {
        out[i] = a.elements().get(i).asInt();
      }
      return out;
    }
    throw new IllegalArgumentException("Expected an int array, got " + this);
  }

  default float[] toFloatArray() {
    if (this instanceof UFloatArray a) {
      return a.values();
    }
    if (this instanceof UArray a) {
      float[] out = new float[a.elements().size()];
      for (int i = 0; i < out.length; i++) {
        out[i] = (float) a.elements().get(i).asDouble();
      }
      return out;
    }
    throw new IllegalArgumentException("Expected a float array, got " + this);
  }

  record UObject(Map<String, UValue> fields) implements UValue {}

  record UArray(List<UValue> elements) implements UValue {}

  record UString(String value) implements UValue {}

  record UNumber(double value) implements UValue {}

  record UBool(boolean value) implements UValue {}

  /** Typed-container fast paths so large numeric tree arrays avoid per-element boxing. */
  record UIntArray(int[] values) implements UValue {}

  record UFloatArray(float[] values) implements UValue {}

  final class UNull implements UValue {
    public static final UNull INSTANCE = new UNull();

    private UNull() {}
  }
}
