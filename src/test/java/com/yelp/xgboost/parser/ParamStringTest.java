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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import com.yelp.xgboost.parser.UValue.UBool;
import com.yelp.xgboost.parser.UValue.UNumber;
import com.yelp.xgboost.parser.UValue.UObject;
import com.yelp.xgboost.parser.UValue.UString;
import java.util.Map;
import org.junit.Test;

/**
 * Pins {@link XGBoostModelParser#paramString}. XGBoost's JSON/UBJSON encodings serialize
 * learner_model_param and tree_param values as strings, but a hand-written or non-standard producer
 * could emit a bare number. The helper coerces a {@link UNumber} to its decimal string so numeric
 * params still load, and rejects any other node type with a typed error rather than silently
 * mis-reading it.
 */
public class ParamStringTest {

  private static UObject obj(String key, UValue value) {
    return new UObject(Map.of(key, value));
  }

  @Test
  public void returnsStringValueDirectly() {
    assertEquals(
        "1.5", XGBoostModelParser.paramString(obj("base_score", new UString("1.5")), "base_score"));
  }

  @Test
  public void coercesNumberToDecimalString() {
    assertEquals(
        "3.0", XGBoostModelParser.paramString(obj("num_class", new UNumber(3.0)), "num_class"));
  }

  @Test
  public void rejectsNonStringNonNumberValue() {
    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> XGBoostModelParser.paramString(obj("flag", new UBool(true)), "flag"));
    assertEquals("the rejection names the offending key", true, e.getMessage().contains("flag"));
  }
}
