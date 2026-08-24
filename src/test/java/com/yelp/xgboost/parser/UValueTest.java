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

import com.yelp.xgboost.parser.UValue.UArray;
import com.yelp.xgboost.parser.UValue.UNumber;
import com.yelp.xgboost.parser.UValue.UObject;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.Test;

/**
 * Direct tests for the {@link UValue} accessor contract: the primitive-array coercions must accept
 * both the typed fast-path nodes and the generic {@link UArray} fallback, {@code apply} must throw
 * on a missing key, and {@code get} must return empty on a non-object.
 */
public class UValueTest {

  @Test
  public void toIntArrayCoercesGenericArray() {
    UValue generic = new UArray(List.of(new UNumber(1), new UNumber(2), new UNumber(3)));
    assertArrayEquals(new int[] {1, 2, 3}, generic.toIntArray());
  }

  @Test
  public void toFloatArrayCoercesGenericArray() {
    UValue generic = new UArray(List.of(new UNumber(0.5), new UNumber(-1.5)));
    assertArrayEquals(new float[] {0.5f, -1.5f}, generic.toFloatArray(), 0.0f);
  }

  @Test
  public void toIntArrayRejectsNonArray() {
    assertThrows(IllegalArgumentException.class, () -> new UNumber(1).toIntArray());
  }

  @Test
  public void toFloatArrayRejectsNonArray() {
    assertThrows(IllegalArgumentException.class, () -> new UNumber(1).toFloatArray());
  }

  @Test
  public void applyThrowsOnMissingKey() {
    UValue obj = new UObject(Map.of("present", new UNumber(1)));
    assertThrows(NoSuchElementException.class, () -> obj.apply("absent"));
  }

  @Test
  public void getReturnsEmptyOnNonObject() {
    assertFalse(new UNumber(1).get("x").isPresent());
  }

  @Test
  public void getReturnsValueOnObject() {
    UValue obj = new UObject(Map.of("k", new UNumber(42)));
    assertTrue(obj.get("k").isPresent());
    assertEquals(42, obj.get("k").get().asInt());
  }
}
