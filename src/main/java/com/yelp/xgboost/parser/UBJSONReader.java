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

import com.yelp.xgboost.parser.UValue.UArray;
import com.yelp.xgboost.parser.UValue.UBool;
import com.yelp.xgboost.parser.UValue.UFloatArray;
import com.yelp.xgboost.parser.UValue.UIntArray;
import com.yelp.xgboost.parser.UValue.UNull;
import com.yelp.xgboost.parser.UValue.UNumber;
import com.yelp.xgboost.parser.UValue.UObject;
import com.yelp.xgboost.parser.UValue.UString;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hand-rolled decoder for XGBoost's UBJSON model encoding (src/common/json.cc UBJWriter). No
 * maintained JVM UBJSON library exists, and XGBoost's codec deviates from the stock spec in two
 * ways this decoder handles:
 *
 * <ul>
 *   <li>Every length/count is framed with the int64 marker {@code 'L'} (8 bytes), never a smaller
 *       int type.
 *   <li>Object keys are bare length-prefixed strings ({@code 'L'} len bytes) with NO leading {@code
 *       'S'} marker.
 *   <li>Generic arrays use {@code '#'} count WITHOUT a preceding {@code '$'}, which stock UBJSON
 *       disallows.
 * </ul>
 *
 * <p>All multi-byte integers and floats are big-endian (network order), matching the JVM default.
 * Homogeneous numeric arrays ({@code [$<type>#L<count>}) decode into primitive {@link UIntArray}/
 * {@link UFloatArray} to avoid boxing the large per-node tree arrays.
 */
final class UBJSONReader {

  private UBJSONReader() {}

  public static UValue read(byte[] bytes) {
    ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN);
    return readValue(buf, buf.get());
  }

  private static UValue readValue(ByteBuffer buf, byte marker) {
    switch ((char) marker) {
      case 'Z':
        return UNull.INSTANCE;
      case 'T':
        return new UBool(true);
      case 'F':
        return new UBool(false);
      case 'i':
        return new UNumber(buf.get());
      case 'U':
        return new UNumber(buf.get() & 0xFF);
      case 'I':
        return new UNumber(buf.getShort());
      case 'l':
        return new UNumber(buf.getInt());
      case 'L':
        return new UNumber(buf.getLong());
      case 'd':
        return new UNumber(buf.getFloat());
      case 'D':
        return new UNumber(buf.getDouble());
      case 'S':
        return new UString(readString(buf));
      case '{':
        return readObject(buf);
      case '[':
        return readArray(buf);
      default:
        throw new IllegalArgumentException(
            "Unsupported UBJSON marker: '"
                + (char) marker
                + "' (0x"
                + Integer.toHexString(marker & 0xFF)
                + ")");
    }
  }

  /**
   * A scalar string: 'S' already consumed by the caller, now 'L' &lt;len&gt; &lt;utf8 bytes&gt;.
   */
  private static String readString(ByteBuffer buf) {
    byte lengthMarker = buf.get();
    if (lengthMarker != 'L') {
      throw new IllegalArgumentException(
          "Expected 'L' string length marker, got '" + (char) lengthMarker + "'");
    }
    return readBytesAsString(buf, buf.getLong());
  }

  /** An object key: bare 'L' &lt;len&gt; &lt;utf8 bytes&gt; with no 'S' marker. */
  private static String readKey(ByteBuffer buf, byte lengthMarker) {
    if (lengthMarker != 'L') {
      throw new IllegalArgumentException(
          "Expected 'L' key length marker, got '" + (char) lengthMarker + "'");
    }
    return readBytesAsString(buf, buf.getLong());
  }

  private static String readBytesAsString(ByteBuffer buf, long length) {
    byte[] dst = new byte[(int) length];
    buf.get(dst);
    return new String(dst, StandardCharsets.UTF_8);
  }

  private static UObject readObject(ByteBuffer buf) {
    Map<String, UValue> fields = new LinkedHashMap<>();
    byte marker = buf.get();
    while (marker != '}') {
      String key = readKey(buf, marker);
      fields.put(key, readValue(buf, buf.get()));
      marker = buf.get();
    }
    return new UObject(fields);
  }

  private static UValue readArray(ByteBuffer buf) {
    byte first = buf.get();
    switch ((char) first) {
      case '$':
        return readTypedArray(buf);
      case '#':
        return readCountedArray(buf);
      default:
        return readUnboundedArray(buf, first);
    }
  }

  /** [$&lt;type&gt;#L&lt;count&gt; then count raw big-endian values with no per-element markers. */
  private static UValue readTypedArray(ByteBuffer buf) {
    char typeMarker = (char) buf.get();
    byte countMarker = buf.get();
    if (countMarker != '#') {
      throw new IllegalArgumentException(
          "Expected '#' after typed-array type, got '" + (char) countMarker + "'");
    }
    byte lengthMarker = buf.get();
    if (lengthMarker != 'L') {
      throw new IllegalArgumentException(
          "Expected 'L' count marker, got '" + (char) lengthMarker + "'");
    }
    int count = (int) buf.getLong();

    switch (typeMarker) {
      case 'l':
        {
          int[] values = new int[count];
          for (int i = 0; i < count; i++) {
            values[i] = buf.getInt();
          }
          return new UIntArray(values);
        }
      case 'd':
        {
          float[] values = new float[count];
          for (int i = 0; i < count; i++) {
            values[i] = buf.getFloat();
          }
          return new UFloatArray(values);
        }
      case 'i':
        {
          int[] values = new int[count];
          for (int i = 0; i < count; i++) {
            values[i] = buf.get();
          }
          return new UIntArray(values);
        }
      case 'U':
        {
          int[] values = new int[count];
          for (int i = 0; i < count; i++) {
            values[i] = buf.get() & 0xFF;
          }
          return new UIntArray(values);
        }
      case 'I':
        {
          int[] values = new int[count];
          for (int i = 0; i < count; i++) {
            values[i] = buf.getShort();
          }
          return new UIntArray(values);
        }
      case 'L':
        {
          List<UValue> elements = new ArrayList<>(count);
          for (int i = 0; i < count; i++) {
            elements.add(new UNumber(buf.getLong()));
          }
          return new UArray(elements);
        }
      case 'D':
        {
          float[] values = new float[count];
          for (int i = 0; i < count; i++) {
            values[i] = (float) buf.getDouble();
          }
          return new UFloatArray(values);
        }
      default:
        throw new IllegalArgumentException(
            "Unsupported typed-array element marker: '" + typeMarker + "'");
    }
  }

  /** '[' '#' 'L' &lt;count&gt; then count fully-typed values. */
  private static UArray readCountedArray(ByteBuffer buf) {
    byte lengthMarker = buf.get();
    if (lengthMarker != 'L') {
      throw new IllegalArgumentException(
          "Expected 'L' count marker, got '" + (char) lengthMarker + "'");
    }
    int count = (int) buf.getLong();
    List<UValue> elements = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      elements.add(readValue(buf, buf.get()));
    }
    return new UArray(elements);
  }

  /** Reader-only fallback: self-described values until ']'. First value marker already read. */
  private static UArray readUnboundedArray(ByteBuffer buf, byte firstMarker) {
    List<UValue> elements = new ArrayList<>();
    byte marker = firstMarker;
    while (marker != ']') {
      elements.add(readValue(buf, marker));
      marker = buf.get();
    }
    return new UArray(elements);
  }
}
