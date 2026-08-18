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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.yelp.xgboost.parser.UValue.UBool;
import com.yelp.xgboost.parser.UValue.UNull;
import com.yelp.xgboost.parser.UValue.UString;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Test;

/**
 * Direct unit tests for the hand-rolled UBJSON decoder, exercising each scalar marker, the XGBoost
 * codec quirks (int64-framed lengths, bare object keys with no 'S', generic arrays with '#' and no
 * '$'), the typed-array primitive fast paths, and every malformed-input error branch. Bytes are
 * written with {@link Ubj}, which mirrors XGBoost's src/common/json.cc UBJWriter exactly.
 */
public class UBJSONReaderTest {

  @Test
  public void decodesScalarIntegerMarkers() {
    assertEquals(-5, (long) UBJSONReader.read(new Ubj().i8(-5).bytes()).asLong());
    assertEquals(200, (long) UBJSONReader.read(new Ubj().u8(200).bytes()).asLong());
    assertEquals(-1234, (long) UBJSONReader.read(new Ubj().i16(-1234).bytes()).asLong());
    assertEquals(70000, (long) UBJSONReader.read(new Ubj().i32(70000).bytes()).asLong());
    assertEquals(5_000_000_000L, UBJSONReader.read(new Ubj().i64(5_000_000_000L).bytes()).asLong());
  }

  @Test
  public void decodesScalarFloatMarkers() {
    assertEquals(1.5, UBJSONReader.read(new Ubj().f32(1.5f).bytes()).asDouble(), 0.0);
    assertEquals(2.25, UBJSONReader.read(new Ubj().f64(2.25).bytes()).asDouble(), 0.0);
  }

  @Test
  public void decodesBooleanAndNullMarkers() {
    assertTrue(((UBool) UBJSONReader.read(new Ubj().t().bytes())).value());
    assertFalse(((UBool) UBJSONReader.read(new Ubj().f().bytes())).value());
    assertEquals(UNull.INSTANCE, UBJSONReader.read(new Ubj().z().bytes()));
  }

  @Test
  public void decodesScalarString() {
    UValue value = UBJSONReader.read(new Ubj().string("gbtree").bytes());
    assertEquals("gbtree", ((UString) value).value());
  }

  @Test
  public void decodesObjectWithBareKeys() {
    byte[] bytes =
        new Ubj()
            .beginObject()
            .key("name")
            .string("gbtree")
            .key("count")
            .i32(3)
            .endObject()
            .bytes();

    UValue root = UBJSONReader.read(bytes);
    assertEquals("gbtree", root.apply("name").str());
    assertEquals(3, root.apply("count").asInt());
  }

  @Test
  public void decodesTypedIntArrayToPrimitiveFastPath() {
    byte[] bytes = new Ubj().typedIntArray(new int[] {-1, 0, 7, 42}).bytes();
    assertArrayEquals(new int[] {-1, 0, 7, 42}, UBJSONReader.read(bytes).toIntArray());
  }

  @Test
  public void decodesTypedFloatArrayToPrimitiveFastPath() {
    byte[] bytes = new Ubj().typedFloatArray(new float[] {0.5f, -1.25f, 3.0f}).bytes();
    assertArrayEquals(
        new float[] {0.5f, -1.25f, 3.0f}, UBJSONReader.read(bytes).toFloatArray(), 0.0f);
  }

  @Test
  public void decodesGenericCountedArray() {
    byte[] bytes = new Ubj().beginCountedArray(2).i32(10).string("x").end().bytes();
    UValue arr = UBJSONReader.read(bytes);
    assertEquals(10, arr.arr().get(0).asInt());
    assertEquals("x", arr.arr().get(1).str());
  }

  @Test
  public void decodesEachTypedArrayElementMarker() {
    assertArrayEquals(
        new int[] {-3, 5},
        UBJSONReader.read(new Ubj().typedInt8Array(new int[] {-3, 5}).bytes()).toIntArray());
    assertArrayEquals(
        new int[] {7, 250},
        UBJSONReader.read(new Ubj().typedUint8Array(new int[] {7, 250}).bytes()).toIntArray());
    assertArrayEquals(
        new int[] {-1000, 1000},
        UBJSONReader.read(new Ubj().typedInt16Array(new int[] {-1000, 1000}).bytes()).toIntArray());
    assertArrayEquals(
        new float[] {1.5f, -2.5f},
        UBJSONReader.read(new Ubj().typedDoubleArray(new double[] {1.5, -2.5}).bytes())
            .toFloatArray(),
        0.0f);

    UValue longs =
        UBJSONReader.read(new Ubj().typedInt64Array(new long[] {5_000_000_000L}).bytes());
    assertEquals(5_000_000_000L, longs.arr().get(0).asLong());
  }

  @Test
  public void decodesUnboundedArrayFallback() {
    byte[] bytes = new Ubj().beginUnboundedArray().i32(4).string("y").endArray().bytes();
    UValue arr = UBJSONReader.read(bytes);
    assertEquals(4, arr.arr().get(0).asInt());
    assertEquals("y", arr.arr().get(1).str());
  }

  @Test
  public void nestsObjectsAndArrays() {
    byte[] bytes =
        new Ubj()
            .beginObject()
            .key("learner")
            .beginObject()
            .key("trees")
            .typedIntArray(new int[] {1, 2})
            .endObject()
            .endObject()
            .bytes();

    UValue root = UBJSONReader.read(bytes);
    assertArrayEquals(new int[] {1, 2}, root.apply("learner").apply("trees").toIntArray());
  }

  @Test
  public void rejectsUnsupportedTopLevelMarker() {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class, () -> UBJSONReader.read(new byte[] {(byte) 'X'}));
    assertTrue(e.getMessage(), e.getMessage().contains("Unsupported UBJSON marker"));
  }

  @Test
  public void rejectsStringWithoutLengthMarker() {
    byte[] bytes = new byte[] {(byte) 'S', (byte) 'i', 0x03};
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> UBJSONReader.read(bytes));
    assertTrue(e.getMessage(), e.getMessage().contains("string length marker"));
  }

  @Test
  public void rejectsObjectKeyWithoutLengthMarker() {
    byte[] bytes = new byte[] {(byte) '{', (byte) 'S', 0x00};
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> UBJSONReader.read(bytes));
    assertTrue(e.getMessage(), e.getMessage().contains("key length marker"));
  }

  @Test
  public void rejectsTypedArrayWithoutCountMarker() {
    byte[] bytes = new byte[] {(byte) '[', (byte) '$', (byte) 'l', (byte) 'X'};
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> UBJSONReader.read(bytes));
    assertTrue(e.getMessage(), e.getMessage().contains("'#'"));
  }

  @Test
  public void rejectsUnsupportedTypedArrayElementMarker() {
    byte[] bytes = new Ubj().beginTypedArrayHeader('X', 0).bytes();
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> UBJSONReader.read(bytes));
    assertTrue(e.getMessage(), e.getMessage().contains("typed-array element marker"));
  }

  /**
   * Byte builder mirroring XGBoost's UBJWriter: every length/count is an int64 framed with 'L',
   * object keys are bare length-prefixed strings with no 'S', big-endian throughout.
   */
  private static final class Ubj {
    private final ByteArrayOutputStream out = new ByteArrayOutputStream();

    byte[] bytes() {
      return out.toByteArray();
    }

    Ubj i8(int v) {
      out.write('i');
      out.write(v & 0xFF);
      return this;
    }

    Ubj u8(int v) {
      out.write('U');
      out.write(v & 0xFF);
      return this;
    }

    Ubj i16(int v) {
      out.write('I');
      writeBig(v, 2);
      return this;
    }

    Ubj i32(int v) {
      out.write('l');
      writeBig(v, 4);
      return this;
    }

    Ubj i64(long v) {
      out.write('L');
      writeBig(v, 8);
      return this;
    }

    Ubj f32(float v) {
      out.write('d');
      writeBig(Float.floatToIntBits(v), 4);
      return this;
    }

    Ubj f64(double v) {
      out.write('D');
      writeBig(Double.doubleToLongBits(v), 8);
      return this;
    }

    Ubj t() {
      out.write('T');
      return this;
    }

    Ubj f() {
      out.write('F');
      return this;
    }

    Ubj z() {
      out.write('Z');
      return this;
    }

    Ubj string(String s) {
      out.write('S');
      len(s);
      return this;
    }

    Ubj beginObject() {
      out.write('{');
      return this;
    }

    Ubj endObject() {
      out.write('}');
      return this;
    }

    Ubj key(String k) {
      len(k);
      return this;
    }

    Ubj beginCountedArray(int count) {
      out.write('[');
      out.write('#');
      out.write('L');
      writeBig(count, 8);
      return this;
    }

    Ubj end() {
      return this;
    }

    Ubj typedIntArray(int[] values) {
      beginTypedArrayHeader('l', values.length);
      for (int v : values) {
        writeBig(v, 4);
      }
      return this;
    }

    Ubj typedFloatArray(float[] values) {
      beginTypedArrayHeader('d', values.length);
      for (float v : values) {
        writeBig(Float.floatToIntBits(v), 4);
      }
      return this;
    }

    Ubj typedInt8Array(int[] values) {
      beginTypedArrayHeader('i', values.length);
      for (int v : values) {
        out.write(v & 0xFF);
      }
      return this;
    }

    Ubj typedUint8Array(int[] values) {
      beginTypedArrayHeader('U', values.length);
      for (int v : values) {
        out.write(v & 0xFF);
      }
      return this;
    }

    Ubj typedInt16Array(int[] values) {
      beginTypedArrayHeader('I', values.length);
      for (int v : values) {
        writeBig(v, 2);
      }
      return this;
    }

    Ubj typedInt64Array(long[] values) {
      beginTypedArrayHeader('L', values.length);
      for (long v : values) {
        writeBig(v, 8);
      }
      return this;
    }

    Ubj typedDoubleArray(double[] values) {
      beginTypedArrayHeader('D', values.length);
      for (double v : values) {
        writeBig(Double.doubleToLongBits(v), 8);
      }
      return this;
    }

    Ubj beginUnboundedArray() {
      out.write('[');
      return this;
    }

    Ubj endArray() {
      out.write(']');
      return this;
    }

    Ubj beginTypedArrayHeader(char type, int count) {
      out.write('[');
      out.write('$');
      out.write(type);
      out.write('#');
      out.write('L');
      writeBig(count, 8);
      return this;
    }

    private void len(String s) {
      byte[] utf8 = s.getBytes(StandardCharsets.UTF_8);
      out.write('L');
      writeBig(utf8.length, 8);
      out.writeBytes(utf8);
    }

    private void writeBig(long value, int numBytes) {
      for (int i = numBytes - 1; i >= 0; i--) {
        out.write((int) ((value >> (8 * i)) & 0xFF));
      }
    }
  }
}
