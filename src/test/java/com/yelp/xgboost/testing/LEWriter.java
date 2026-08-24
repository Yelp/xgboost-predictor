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
package com.yelp.xgboost.testing;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Little-endian byte builder for hand-packing synthetic legacy-binary XGBoost models in tests. The
 * legacy {@link com.yelp.xgboost.parser.ModelReader} path reads primitives little-endian and
 * length-prefixes strings with an int64, which {@link #str} mirrors.
 */
public final class LEWriter {
  private final ByteArrayOutputStream out = new ByteArrayOutputStream();

  public LEWriter bytes(byte[] b) {
    out.writeBytes(b);
    return this;
  }

  public LEWriter i(int value) {
    out.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    return this;
  }

  public LEWriter l(long value) {
    out.writeBytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array());
    return this;
  }

  public LEWriter f(float value) {
    out.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(value).array());
    return this;
  }

  public LEWriter str(String value) {
    byte[] utf8 = value.getBytes(StandardCharsets.UTF_8);
    l(utf8.length);
    return bytes(utf8);
  }

  public LEWriter zeros(int count) {
    out.writeBytes(new byte[count]);
    return this;
  }

  public byte[] toByteArray() {
    return out.toByteArray();
  }
}
