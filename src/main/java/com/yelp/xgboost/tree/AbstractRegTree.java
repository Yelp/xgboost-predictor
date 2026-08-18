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
import com.yelp.xgboost.parser.ModelReader;
import java.io.IOException;
import java.io.Serializable;

/**
 * Provides basic interface and common functionality for a binary decision tree to be loaded and
 * evaluated.
 */
public abstract class AbstractRegTree implements Serializable {
  /**
   * Loads the model from a provided ModelReader
   *
   * @param reader
   * @throws IOException
   */
  public final void loadModel(ModelReader reader) throws IOException {
    loadModel(new Param(reader));
  }

  /**
   * Loads the model from a parameters instance
   *
   * @param param
   */
  public abstract void loadModel(Param param);

  protected int getRootNode() {
    return 0;
  }

  protected abstract int getNextNode(int node, FVec feat);

  protected abstract boolean isLeafNode(int node);

  protected abstract float getLeafValue(int node);

  protected abstract int getLeafIndex(int node);

  /**
   * Returns the leaf node index for the given fvec starting at the tree's root
   *
   * @param feat feature vector to evaluate tree on
   * @return leaf node index
   */
  public final int getLeafIndex(FVec feat) {
    return getLeafIndex(getLeafNodeForFeat(feat, getRootNode()));
  }

  /**
   * Returns the leaf node index for the given fvec starting at the provided node
   *
   * @param feat feature vector to test tree on
   * @param node first node considered for evaluation
   * @return leaf node index
   */
  private int getLeafNodeForFeat(FVec feat, int node) {
    while (!isLeafNode(node)) {
      node = getNextNode(node, feat);
    }

    return node;
  }

  /**
   * Returns the leaf node value for the given fvec starting at the tree's root
   *
   * @param feat feature vector to evaluate tree on
   * @return leaf node index
   */
  public final float getLeafValue(FVec feat) {
    return getLeafValue(getLeafNodeForFeat(feat, getRootNode()));
  }

  /** Parameters. */
  public static class Param implements Serializable {
    /*! \brief number of start root */
    final int num_roots;
    /*! \brief total number of nodes */
    final int num_nodes;
    /*!\brief number of deleted nodes */
    final int num_deleted;
    /*! \brief maximum depth, this is a statistics of the tree */
    final int max_depth;
    /*! \brief  number of features used for tree construction */
    final int num_feature;
    /*!
     * \brief leaf vector size, used for vector tree
     * used to store more than one dimensional information in tree
     */
    final int size_leaf_vector;
    /*! \brief reserved part */
    final int[] reserved;

    public final Node[] nodeInfo;

    public Param(ModelReader reader) throws IOException {
      num_roots = reader.readInt();
      num_nodes = reader.readInt();
      num_deleted = reader.readInt();
      max_depth = reader.readInt();
      num_feature = reader.readInt();

      size_leaf_vector = reader.readInt();
      reserved = reader.readIntArray(31);

      nodeInfo = new Node[num_nodes];

      for (int i = 0; i < num_nodes; i++) {
        nodeInfo[i] = new Node(i, reader);
      }

      for (int i = 0; i < num_nodes; i++) {
        nodeInfo[i].readStats(reader);
      }
    }

    /**
     * Builds a tree from the parallel node arrays used by XGBoost's JSON/UBJSON model format (see
     * io_utils.h tree_field). Leaf nodes have leftChildren[i] == -1 and hold their output value in
     * splitConditions[i]. sumHessian is required for cover-based reordering (PreorderRegTree);
     * isCategorical/categories are null for purely numeric trees.
     */
    public Param(
        int numNodes,
        int[] leftChildren,
        int[] rightChildren,
        int[] splitIndices,
        float[] splitConditions,
        boolean[] defaultLeft,
        float[] sumHessian,
        boolean[] isCategorical,
        int[][] categories) {
      num_roots = 1;
      num_nodes = numNodes;
      num_deleted = 0;
      max_depth = 0;
      num_feature = 0;
      size_leaf_vector = 0;
      reserved = new int[31];

      nodeInfo = new Node[numNodes];
      for (int i = 0; i < numNodes; i++) {
        nodeInfo[i] =
            new Node(
                i,
                leftChildren[i],
                rightChildren[i],
                splitIndices[i],
                defaultLeft[i],
                splitConditions[i],
                sumHessian == null ? 0f : sumHessian[i],
                isCategorical != null && isCategorical[i],
                categories == null ? null : categories[i]);
      }
    }
  }

  /** Stores attributes of a tree node. Later it is transformed to int[] array. */
  public static class Node implements Serializable {
    final int id;
    // pointer to parent, highest bit is used to
    // indicate whether it's a left child or not
    final int parent_;
    // pointer to left, right
    int cleft_, cright_;
    // split feature index, left split or right split depends on the highest bit
    /* unsigned */ int sindex_;
    // extra info (leaf_value or split_cond)
    float leaf_value;
    float split_cond;

    int _defaultNext;
    int _splitIndex;
    boolean _isLeaf;

    /*! \brief whether this node splits on a categorical feature (JSON/UBJSON models only) */
    public boolean _isCategorical;
    /*! \brief the set of category values routed right; null for numeric splits */
    int[] _categories;

    /*! \brief loss change caused by current split */
    float loss_chg;
    /*! \brief sum of hessian values, used to measure coverage of data */
    float sum_hess;
    /*! \brief weight of current node */
    float base_weight;
    /*! \brief number of child that is leaf node known up to now */
    int leaf_child_cnt;

    Node(int id, ModelReader reader) throws IOException {
      this.id = id;

      parent_ = reader.readInt();
      cleft_ = reader.readInt();
      cright_ = reader.readInt();
      sindex_ = reader.readInt();

      if (is_leaf()) {
        leaf_value = reader.readFloat();
        split_cond = Float.NaN;
      } else {
        split_cond = reader.readFloat();
        leaf_value = Float.NaN;
      }

      _defaultNext = cdefault();
      _splitIndex = split_index();
      _isLeaf = is_leaf();
    }

    /**
     * Builds a node from already-decoded JSON/UBJSON fields. sindex_ packs the split feature index
     * with the default-left bit in the high bit, matching the binary format so split_index() and
     * default_left() keep working unchanged.
     */
    Node(
        int id,
        int cleft,
        int cright,
        int splitIndex,
        boolean defaultLeft,
        float splitConditionOrLeafValue,
        float sumHessian,
        boolean isCategorical,
        int[] categories) {
      this.id = id;
      this.parent_ = -1;
      this.cleft_ = cleft;
      this.cright_ = cright;
      this.sindex_ = (splitIndex & ((1 << 31) - 1)) | (defaultLeft ? (1 << 31) : 0);

      if (is_leaf()) {
        this.leaf_value = splitConditionOrLeafValue;
        this.split_cond = Float.NaN;
      } else {
        this.split_cond = splitConditionOrLeafValue;
        this.leaf_value = Float.NaN;
      }

      this.sum_hess = sumHessian;
      this._isCategorical = isCategorical;
      this._categories = categories;

      this._defaultNext = cdefault();
      this._splitIndex = split_index();
      this._isLeaf = is_leaf();
    }

    void readStats(ModelReader reader) throws IOException {
      loss_chg = reader.readFloat();
      sum_hess = reader.readFloat();
      base_weight = reader.readFloat();
      leaf_child_cnt = reader.readInt();
    }

    boolean is_leaf() {
      return cleft_ == -1;
    }

    int split_index() {
      return (int) (sindex_ & ((1l << 31) - 1l));
    }

    int cdefault() {
      return default_left() ? cleft_ : cright_;
    }

    boolean default_left() {
      return (sindex_ >>> 31) != 0;
    }

    int next(FVec feat) {
      Float fvalue = feat.fvalue(_splitIndex);
      if (fvalue == null || fvalue.isNaN()) {
        return _defaultNext;
      }
      if (_isCategorical) {
        return categoricalMember(fvalue) ? cright_ : cleft_;
      }
      return (fvalue < split_cond) ? cleft_ : cright_;
    }

    /**
     * Category membership test matching XGBoost's common::Decision (categorical.h): a value in the
     * split's category set routes right; a non-member, negative, non-integral, or out-of-range (>=
     * 16777216) value routes left, as one-hot encoding would.
     */
    private boolean categoricalMember(float fvalue) {
      if (_categories == null
          || fvalue < 0f
          || fvalue >= 16777216f
          || fvalue != Math.floor(fvalue)) {
        return false;
      }
      int cat = (int) fvalue;
      for (int c : _categories) {
        if (c == cat) {
          return true;
        }
      }
      return false;
    }
  }
}
