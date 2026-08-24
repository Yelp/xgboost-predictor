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

import com.yelp.xgboost.Predictor;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;

/**
 * Builds a {@link Predictor} from a serialized XGBoost model, dispatching on the model format so
 * both existing legacy-binary bundles and new 3.3.0 UBJSON bundles load through the same pure-JVM
 * engine.
 *
 * <p>Dispatch mirrors XGBoost's own DispatchModelType (c_api.cc): a leading &#123; byte marks a
 * UBJSON document (XGBoost's UBJSON object also opens with &#123;); anything else is the legacy
 * pre-1.0 binary struct dump (whose first bytes are the "binf" magic or a little-endian float
 * base_score). These two branches cover both the legacy binary format (xgboost &lt;= 2.0.3) and the
 * current UBJSON format written by {@code Booster.saveModel}.
 *
 * <p>Plain-text JSON is not accepted: {@link UBJSONReader} decodes XGBoost's UBJSON codec, not
 * textual JSON, and MLeap's write path never emits it. A textual JSON model would fail fast in the
 * reader rather than mis-parse.
 */
public final class PredictorFactory {

  private static final byte JSON_OBJECT_MARKER = '{';

  private PredictorFactory() {}

  public static Predictor fromModelStream(InputStream in) {
    return fromModelBytes(readAll(in));
  }

  public static Predictor fromModelBytes(byte[] bytes) {
    if (bytes.length == 0) {
      throw new IllegalArgumentException("Cannot load an XGBoost model from empty bytes");
    }
    Predictor predictor;
    if (bytes[0] == JSON_OBJECT_MARKER) {
      predictor = XGBoostModelParser.parse(UBJSONReader.read(bytes));
    } else {
      predictor = legacyPredictor(bytes);
    }
    predictor.setRawModel(bytes);
    return predictor;
  }

  private static Predictor legacyPredictor(byte[] bytes) {
    try {
      return new Predictor(new ByteArrayInputStream(bytes));
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static byte[] readAll(InputStream in) {
    try (in) {
      return in.readAllBytes();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }
}
