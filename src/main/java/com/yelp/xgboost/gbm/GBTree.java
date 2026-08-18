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
package com.yelp.xgboost.gbm;

import com.yelp.xgboost.FVec;
import com.yelp.xgboost.parser.ModelReader;
import com.yelp.xgboost.tree.AbstractRegTree;
import com.yelp.xgboost.tree.CategoricalRegTree;
import com.yelp.xgboost.tree.PreorderRegTree;
import java.io.IOException;
import java.io.Serializable;
import java.util.function.Function;

/** Gradient boosted tree implementation. */
public class GBTree extends GBBase {
  private interface TreeCreationStrategy
      extends Function<AbstractRegTree.Param, AbstractRegTree>, Serializable {}

  private ModelParam mparam;
  private AbstractRegTree[] trees;
  private TreeCreationStrategy treeCreationStrategy;

  private AbstractRegTree[][] _groupTrees;

  public GBTree() {
    // Categorical trees traverse the object node representation directly; purely numeric trees use
    // the cache-optimized primitive-array PreorderRegTree.
    this(param -> hasCategoricalSplit(param) ? new CategoricalRegTree() : new PreorderRegTree());
  }

  public GBTree(TreeCreationStrategy treeCreationStrategy) {
    this.treeCreationStrategy = treeCreationStrategy;
  }

  private static boolean hasCategoricalSplit(AbstractRegTree.Param param) {
    for (AbstractRegTree.Node node : param.nodeInfo) {
      if (node._isCategorical) {
        return true;
      }
    }
    return false;
  }

  /**
   * Builds the booster from trees already decoded from XGBoost's JSON/UBJSON format. treeInfo[t]
   * gives the output group (class) for tree t, mirroring loadModel's tree_info handling.
   */
  public void loadModel(AbstractRegTree.Param[] treeParams, int[] treeInfo) {
    trees = new AbstractRegTree[treeParams.length];
    for (int i = 0; i < treeParams.length; i++) {
      trees[i] = this.treeCreationStrategy.apply(treeParams[i]);
      trees[i].loadModel(treeParams[i]);
    }
    groupTreesByOutput(treeInfo);
  }

  private void groupTreesByOutput(int[] tree_info) {
    _groupTrees = new AbstractRegTree[num_output_group][];
    for (int i = 0; i < num_output_group; i++) {
      int treeCount = 0;
      for (int j = 0; j < tree_info.length; j++) {
        if (tree_info[j] == i) {
          treeCount++;
        }
      }

      _groupTrees[i] = new AbstractRegTree[treeCount];
      treeCount = 0;

      for (int j = 0; j < tree_info.length; j++) {
        if (tree_info[j] == i) {
          _groupTrees[i][treeCount++] = trees[j];
        }
      }
    }
  }

  @Override
  public void loadModel(ModelReader reader, boolean with_pbuffer) throws IOException {
    mparam = new ModelParam(reader);

    trees = new AbstractRegTree[mparam.num_trees];
    for (int i = 0; i < mparam.num_trees; i++) {
      AbstractRegTree.Param param = new AbstractRegTree.Param(reader);
      trees[i] = this.treeCreationStrategy.apply(param);
      trees[i].loadModel(param);
    }

    int[] tree_info = mparam.num_trees > 0 ? reader.readIntArray(mparam.num_trees) : new int[0];

    if (mparam.num_pbuffer != 0 && with_pbuffer) {
      reader.skip(4 * predBufferSize());
      reader.skip(4 * predBufferSize());
    }

    groupTreesByOutput(tree_info);
  }

  @Override
  public float[] predict(FVec feat, int ntree_limit) {
    float[] preds = new float[num_output_group];
    for (int gid = 0; gid < num_output_group; gid++) {
      preds[gid] = pred(feat, gid, ntree_limit);
    }
    return preds;
  }

  @Override
  public float predictSingle(FVec feat, int ntree_limit) {
    if (num_output_group != 1) {
      throw new IllegalStateException(
          "Can't invoke predictSingle() because this model outputs multiple values: "
              + num_output_group);
    }
    return pred(feat, 0, ntree_limit);
  }

  float pred(FVec feat, int bst_group, int ntree_limit) {
    AbstractRegTree[] trees = _groupTrees[bst_group];
    int treeleft = boundedTreeCount(ntree_limit, trees.length);

    float psum = 0;
    for (int i = 0; i < treeleft; i++) {
      psum += trees[i].getLeafValue(feat);
    }

    return psum;
  }

  @Override
  public int[] predictLeaf(FVec feat, int ntree_limit) {
    return predPath(feat, ntree_limit);
  }

  int[] predPath(FVec feat, int ntree_limit) {
    int treeleft = boundedTreeCount(ntree_limit, trees.length);

    int[] leafIndex = new int[treeleft];
    for (int i = 0; i < treeleft; i++) {
      leafIndex[i] = trees[i].getLeafIndex(feat);
    }
    return leafIndex;
  }

  /**
   * Number of trees to score for the given limit. {@code ntree_limit == 0} means all trees;
   * otherwise the limit must not exceed {@code available}. Native xgboost rejects an out-of-range
   * limit ("Out of range for tree layers") rather than clamping, so we throw rather than silently
   * scoring fewer or reading past the array.
   */
  private static int boundedTreeCount(int ntree_limit, int available) {
    if (ntree_limit == 0) {
      return available;
    }
    if (ntree_limit < 0 || ntree_limit > available) {
      throw new IllegalArgumentException(
          "ntree_limit " + ntree_limit + " is out of range for " + available + " trees");
    }
    return ntree_limit;
  }

  private long predBufferSize() {
    return num_output_group * mparam.num_pbuffer * (mparam.size_leaf_vector + 1);
  }

  static class ModelParam implements Serializable {
    /*! \brief number of trees */
    final int num_trees;
    /*! \brief number of root: default 0, means single tree */
    final int num_roots;
    /*! \brief size of predicton buffer allocated used for buffering */
    final long num_pbuffer;
    /*! \brief size of leaf vector needed in tree */
    final int size_leaf_vector;
    /*! \brief reserved space */
    final int[] reserved;

    ModelParam(ModelReader reader) throws IOException {
      num_trees = reader.readInt();
      num_roots = reader.readInt();
      reader.readInt(); // num_feature deprecated
      reader.readInt(); // read padding
      num_pbuffer = reader.readLong();
      reader.readInt(); // num_output_group not used anymore
      size_leaf_vector = reader.readInt();
      reserved = reader.readIntArray(31);
      reader.readInt(); // read padding
    }
  }
}
