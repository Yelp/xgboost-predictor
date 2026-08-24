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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal recursive-descent JSON reader for test fixtures, so the test suite needs no JSON
 * dependency. Supports objects, arrays, strings, numbers, booleans and null, which is all the
 * golden fixtures use. Object field order is preserved.
 */
abstract class Json {

  Obj asObj() {
    return (Obj) this;
  }

  List<Json> asArr() {
    return ((Arr) this).elements;
  }

  String asString() {
    return ((Str) this).value;
  }

  double asDouble() {
    return ((Num) this).value;
  }

  static final class Obj extends Json {
    final Map<String, Json> fields = new LinkedHashMap<>();

    Json get(String key) {
      Json value = fields.get(key);
      if (value == null) {
        throw new IllegalStateException("Missing key: " + key);
      }
      return value;
    }
  }

  static final class Arr extends Json {
    final List<Json> elements = new ArrayList<>();
  }

  static final class Str extends Json {
    final String value;

    Str(String value) {
      this.value = value;
    }
  }

  static final class Num extends Json {
    final double value;

    Num(double value) {
      this.value = value;
    }
  }

  static final class Lit extends Json {}

  static Json parse(String text) {
    Parser parser = new Parser(text);
    parser.skipWhitespace();
    Json value = parser.parseValue();
    parser.skipWhitespace();
    if (!parser.atEnd()) {
      throw new IllegalStateException("Trailing content at index " + parser.pos);
    }
    return value;
  }

  private static final class Parser {
    private final String text;
    private int pos;

    Parser(String text) {
      this.text = text;
    }

    boolean atEnd() {
      return pos >= text.length();
    }

    void skipWhitespace() {
      while (pos < text.length() && Character.isWhitespace(text.charAt(pos))) {
        pos++;
      }
    }

    Json parseValue() {
      char c = text.charAt(pos);
      switch (c) {
        case '{':
          return parseObject();
        case '[':
          return parseArray();
        case '"':
          return new Str(parseString());
        case 't':
          return parseKeyword("true");
        case 'f':
          return parseKeyword("false");
        case 'n':
          return parseKeyword("null");
        default:
          return parseNumber();
      }
    }

    private Obj parseObject() {
      Obj obj = new Obj();
      pos++;
      skipWhitespace();
      if (text.charAt(pos) == '}') {
        pos++;
        return obj;
      }
      while (true) {
        skipWhitespace();
        String key = parseString();
        skipWhitespace();
        expect(':');
        skipWhitespace();
        obj.fields.put(key, parseValue());
        skipWhitespace();
        char c = text.charAt(pos++);
        if (c == '}') {
          return obj;
        }
        if (c != ',') {
          throw new IllegalStateException("Expected ',' or '}' at index " + (pos - 1));
        }
      }
    }

    private Arr parseArray() {
      Arr arr = new Arr();
      pos++;
      skipWhitespace();
      if (text.charAt(pos) == ']') {
        pos++;
        return arr;
      }
      while (true) {
        skipWhitespace();
        arr.elements.add(parseValue());
        skipWhitespace();
        char c = text.charAt(pos++);
        if (c == ']') {
          return arr;
        }
        if (c != ',') {
          throw new IllegalStateException("Expected ',' or ']' at index " + (pos - 1));
        }
      }
    }

    private String parseString() {
      expect('"');
      StringBuilder sb = new StringBuilder();
      while (true) {
        char c = text.charAt(pos++);
        if (c == '"') {
          return sb.toString();
        }
        if (c == '\\') {
          char escaped = text.charAt(pos++);
          switch (escaped) {
            case '"':
              sb.append('"');
              break;
            case '\\':
              sb.append('\\');
              break;
            case '/':
              sb.append('/');
              break;
            case 'n':
              sb.append('\n');
              break;
            case 't':
              sb.append('\t');
              break;
            case 'r':
              sb.append('\r');
              break;
            case 'b':
              sb.append('\b');
              break;
            case 'f':
              sb.append('\f');
              break;
            case 'u':
              sb.append((char) Integer.parseInt(text.substring(pos, pos + 4), 16));
              pos += 4;
              break;
            default:
              throw new IllegalStateException("Invalid escape \\" + escaped);
          }
        } else {
          sb.append(c);
        }
      }
    }

    private Num parseNumber() {
      int start = pos;
      while (pos < text.length() && "+-.eE0123456789".indexOf(text.charAt(pos)) >= 0) {
        pos++;
      }
      return new Num(Double.parseDouble(text.substring(start, pos)));
    }

    private Lit parseKeyword(String keyword) {
      if (!text.startsWith(keyword, pos)) {
        throw new IllegalStateException("Expected " + keyword + " at index " + pos);
      }
      pos += keyword.length();
      return new Lit();
    }

    private void expect(char c) {
      if (text.charAt(pos++) != c) {
        throw new IllegalStateException("Expected '" + c + "' at index " + (pos - 1));
      }
    }
  }
}
