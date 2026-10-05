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
package com.yelp.xgboost;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

import java.util.Random;
import org.junit.Test;

/**
 * Pins {@link FVec#fromSparse} to {@link FVec#fromArray} and {@link
 * FVec#fromArrayWithZeroAsMissing} over the densified row, index by index, on rows mixing stored
 * values, explicit zeros, NaN, absent indices and out-of-range lookups.
 */
public class FVecSparseTest {

  private static final int ROWS = 200;
  private static final int MAX_WIDTH = 5000;

  @Test
  public void matchesDensifiedRowWhereZeroIsAValue() {
    assertMatchesDensified(false);
  }

  @Test
  public void matchesDensifiedRowWhereZeroIsMissing() {
    assertMatchesDensified(true);
  }

  @Test
  public void emptyRowReadsAsZerosWithinSize() {
    FVec fvec = FVec.fromSparse(new int[0], new float[0], 3, false);
    assertEquals(Float.valueOf(0.0f), fvec.fvalue(2));
    assertNull(fvec.fvalue(3));
    assertNull(FVec.fromSparse(new int[0], new float[0], 3, true).fvalue(0));
  }

  @Test
  public void rejectsMisalignedIndicesAndValues() {
    assertThrows(
        IllegalArgumentException.class,
        () -> FVec.fromSparse(new int[] {0, 1}, new float[] {1.0f}, 2, false));
  }

  private static void assertMatchesDensified(boolean treatsZeroAsNA) {
    Random random = new Random(7L);
    for (int row = 0; row < ROWS; row++) {
      assertRowMatches(SparseRow.random(random), treatsZeroAsNA);
    }
  }

  private static void assertRowMatches(SparseRow row, boolean treatsZeroAsNA) {
    FVec expected =
        treatsZeroAsNA ? FVec.fromArrayWithZeroAsMissing(row.dense()) : FVec.fromArray(row.dense());
    FVec fromFloats = FVec.fromSparse(row.indices, row.values, row.size, treatsZeroAsNA);
    FVec fromDoubles = FVec.fromSparse(row.indices, row.doubles(), row.size, treatsZeroAsNA);
    for (int index = 0; index < row.size + 3; index++) {
      assertEquals("index " + index, expected.fvalue(index), fromFloats.fvalue(index));
      assertEquals("index " + index, expected.fvalue(index), fromDoubles.fvalue(index));
    }
  }

  /** Random sparse row with distinct indices, some explicit zeros and some NaN values. */
  private record SparseRow(int[] indices, float[] values, int size) {

    static SparseRow random(Random random) {
      int size = 1 + random.nextInt(MAX_WIDTH);
      int[] indices = random.ints(0, size).distinct().limit(random.nextInt(size + 1)).toArray();
      float[] values = new float[indices.length];
      for (int i = 0; i < values.length; i++) {
        values[i] = value(random);
      }
      return new SparseRow(indices, values, size);
    }

    private static float value(Random random) {
      int kind = random.nextInt(10);
      if (kind == 0) {
        return 0.0f;
      }
      return kind == 1 ? Float.NaN : (float) random.nextGaussian();
    }

    float[] dense() {
      float[] dense = new float[size];
      for (int i = 0; i < indices.length; i++) {
        dense[indices[i]] = values[i];
      }
      return dense;
    }

    double[] doubles() {
      double[] doubles = new double[values.length];
      for (int i = 0; i < values.length; i++) {
        doubles[i] = values[i];
      }
      return doubles;
    }
  }
}
