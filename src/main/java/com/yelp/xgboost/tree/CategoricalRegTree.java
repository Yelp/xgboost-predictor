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
package com.yelp.xgboost.tree;

import com.yelp.xgboost.FVec;

/**
 * Regression tree that traverses the object node representation directly, supporting both numeric
 * and categorical splits. Unlike {@link PreorderRegTree}, it does not repack the tree into a
 * primitive int array, so it is used only for trees that actually contain categorical splits
 * (rare); purely numeric trees keep using the cache-optimized {@link PreorderRegTree}.
 */
public class CategoricalRegTree extends AbstractRegTree {
  private Node[] nodes;

  @Override
  public void loadModel(Param param) {
    this.nodes = param.nodeInfo;
  }

  @Override
  protected int getNextNode(int node, FVec feat) {
    return nodes[node].next(feat);
  }

  @Override
  protected boolean isLeafNode(int node) {
    return nodes[node]._isLeaf;
  }

  @Override
  protected float getLeafValue(int node) {
    return nodes[node].leaf_value;
  }

  @Override
  protected int getLeafIndex(int node) {
    return node;
  }
}
