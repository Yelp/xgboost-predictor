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
package com.yelp.xgboost.parser;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Reads the legacy-binary XGBoost model from a stream. The UBJSON counterpart is {@link
 * UBJSONReader}. Package-internal: consumed by the boosters and trees while loading a legacy model.
 */
public class ModelReader implements Closeable {
  private final InputStream stream;
  private byte[] buffer;

  public ModelReader(InputStream in) {
    stream = in;
  }

  private int fillBuffer(int numBytes) throws IOException {
    if (buffer == null || buffer.length < numBytes) {
      buffer = new byte[numBytes];
    }

    int numBytesRead = 0;
    while (numBytesRead < numBytes) {
      int count = stream.read(buffer, numBytesRead, numBytes - numBytesRead);
      if (count < 0) {
        return numBytesRead;
      }
      numBytesRead += count;
    }

    return numBytesRead;
  }

  public byte[] readByteArray(int numBytes) throws IOException {
    int numBytesRead = fillBuffer(numBytes);
    if (numBytesRead < numBytes) {
      throw new EOFException(
          String.format(
              "Cannot read byte array (shortage): expected = %d, actual = %d",
              numBytes, numBytesRead));
    }

    byte[] result = new byte[numBytes];
    System.arraycopy(buffer, 0, result, 0, numBytes);

    return result;
  }

  public int readInt() throws IOException {
    int numBytesRead = fillBuffer(4);
    if (numBytesRead < 4) {
      throw new EOFException("Cannot read int value (shortage): " + numBytesRead);
    }

    return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getInt();
  }

  public int[] readIntArray(int numValues) throws IOException {
    int numBytesRead = fillBuffer(numValues * 4);
    if (numBytesRead < numValues * 4) {
      throw new EOFException(
          String.format(
              "Cannot read int array (shortage): expected = %d, actual = %d",
              numValues * 4, numBytesRead));
    }

    ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

    int[] result = new int[numValues];
    for (int i = 0; i < numValues; i++) {
      result[i] = byteBuffer.getInt();
    }

    return result;
  }

  public int readUnsignedInt() throws IOException {
    int result = readInt();
    if (result < 0) {
      throw new IOException("Cannot read unsigned int (overflow): " + result);
    }

    return result;
  }

  public long readLong() throws IOException {
    int numBytesRead = fillBuffer(8);
    if (numBytesRead < 8) {
      throw new IOException("Cannot read long value (shortage): " + numBytesRead);
    }

    return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getLong();
  }

  public float asFloat(byte[] bytes) {
    return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getFloat();
  }

  public float readFloat() throws IOException {
    int numBytesRead = fillBuffer(4);
    if (numBytesRead < 4) {
      throw new IOException("Cannot read float value (shortage): " + numBytesRead);
    }

    return ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN).getFloat();
  }

  public float[] readFloatArray(int numValues) throws IOException {
    int numBytesRead = fillBuffer(numValues * 4);
    if (numBytesRead < numValues * 4) {
      throw new EOFException(
          String.format(
              "Cannot read float array (shortage): expected = %d, actual = %d",
              numValues * 4, numBytesRead));
    }

    ByteBuffer byteBuffer = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN);

    float[] result = new float[numValues];
    for (int i = 0; i < numValues; i++) {
      result[i] = byteBuffer.getFloat();
    }

    return result;
  }

  public void skip(long numBytes) throws IOException {
    long numBytesRead = stream.skip(numBytes);
    if (numBytesRead < numBytes) {
      throw new IOException("Cannot skip bytes: " + numBytesRead);
    }
  }

  public String readString() throws IOException {
    long length = readLong();
    if (length > Integer.MAX_VALUE) {
      throw new IOException("Too long string: " + length);
    }

    return readString((int) length);
  }

  private String readString(int numBytes) throws IOException {
    int numBytesRead = fillBuffer(numBytes);
    if (numBytesRead < numBytes) {
      throw new IOException(
          String.format("Cannot read string(%d) (shortage): %d", numBytes, numBytesRead));
    }

    return new String(buffer, 0, numBytes, StandardCharsets.UTF_8);
  }

  @Override
  public void close() throws IOException {
    stream.close();
  }
}
