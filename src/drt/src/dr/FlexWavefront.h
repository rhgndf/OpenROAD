// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2019-2025, The OpenROAD Authors

#pragma once

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "db/infra/frBox.h"
#include "dr/FlexMazeTypes.h"
#include "frBaseTypes.h"
#include "global.h"

namespace drt {
class FlexWavefrontGrid
{
 public:
  FlexWavefrontGrid(int xIn,
                    int yIn,
                    int zIn,
                    frCoord vLengthXIn,
                    frCoord vLengthYIn,
                    bool prevViaUpIn,
                    frCoord tLengthIn,
                    frCoord distIn,
                    frCost pathCostIn,
                    frCost costIn,
                    const std::bitset<WAVEFRONTBITSIZE>& backTraceBufferIn = {})
      : xIdx_(xIn),
        yIdx_(yIn),
        zIdx_(zIn),
        pathCost_(pathCostIn),
        cost_(costIn),
        vLengthX_(vLengthXIn),
        vLengthY_(vLengthYIn),
        dist_(distIn),
        prevViaUp_(prevViaUpIn),
        tLength_(tLengthIn),
        backTraceBuffer_(backTraceBufferIn)
  {
  }
  bool operator<(const FlexWavefrontGrid& b) const
  {
    if (cost_ != b.cost_) {
      return cost_ > b.cost_;  // prefer smaller cost
    }
    if (dist_ != b.dist_) {
      return dist_ > b.dist_;  // prefer routing close to pin gravity center
                               // (centerPt)
    }
    if (zIdx_ != b.zIdx_) {
      return zIdx_ < b.zIdx_;  // prefer upper layer
    }
    return pathCost_ < b.pathCost_;  // prefer larger pathcost, DFS-style
  }
  // getters
  frMIdx x() const { return xIdx_; }
  frMIdx y() const { return yIdx_; }
  frMIdx z() const { return zIdx_; }
  frCost getPathCost() const { return pathCost_; }
  frCost getCost() const { return cost_; }
  frCoord getDist() const { return dist_; }
  const std::bitset<WAVEFRONTBITSIZE>& getBackTraceBuffer() const
  {
    return backTraceBuffer_;
  }
  frCoord getLength() const { return vLengthX_; }
  void getVLength(frCoord& vLengthXIn, frCoord& vLengthYIn) const
  {
    vLengthXIn = vLengthX_;
    vLengthYIn = vLengthY_;
  }
  bool isPrevViaUp() const { return prevViaUp_; }
  frCoord getTLength() const { return tLength_; }
  frUInt4 getId() const { return id_; }
  frUInt4 getParentId() const { return parent_id_; }
  // setters

  void resetLength()
  {
    vLengthX_ = 0;
    vLengthY_ = 0;
  }
  void setPrevViaUp(bool in) { prevViaUp_ = in; }
  frDirEnum getLastDir() const
  {
    auto currDirVal = backTraceBuffer_.to_ulong() & 0b111u;
    return static_cast<frDirEnum>(currDirVal);
  }
  bool isBufferFull() const
  {
    std::bitset<WAVEFRONTBITSIZE> mask = WAVEFRONTBUFFERHIGHMASK;
    return (mask & backTraceBuffer_).any();
  }
  frDirEnum shiftAddBuffer(const frDirEnum& dir)
  {
    auto retBS = static_cast<frDirEnum>(
        (backTraceBuffer_ >> (WAVEFRONTBITSIZE - DIRBITSIZE)).to_ulong());
    backTraceBuffer_ <<= DIRBITSIZE;
    std::bitset<WAVEFRONTBITSIZE> newBS = (unsigned) dir;
    backTraceBuffer_ |= newBS;
    return retBS;
  }
  void setSrcTaperBox(const frBox3D* b) { srcTaperBox_ = b; }
  const frBox3D* getSrcTaperBox() const { return srcTaperBox_; }
  void setId(const frUInt4 in) { id_ = in; }
  void setParentId(const frUInt4 in) { parent_id_ = in; }

 private:
  frMIdx xIdx_, yIdx_, zIdx_;
  frCost pathCost_;  // path cost
  frCost cost_;      // path + est cost
  frCoord vLengthX_;
  frCoord vLengthY_;
  frCoord dist_;  // to maze center
  bool prevViaUp_;
  frCoord tLength_;  // length since last turn
  std::bitset<WAVEFRONTBITSIZE> backTraceBuffer_;
  const frBox3D* srcTaperBox_ = nullptr;
  frUInt4 id_{0};
  frUInt4 parent_id_{0};
};

class FlexWavefront
{
  // Binary heap over compact 16-byte keys; the (64-byte) grids live in a slot
  // pool with a free list.  The key order is exactly FlexWavefrontGrid's
  // operator< (cost asc, dist asc, z desc, pathCost desc) and the heap uses the
  // same std::push_heap/pop_heap as std::priority_queue, so the pop order is
  // identical to the previous priority_queue<FlexWavefrontGrid>.
  //
  // hi = cost:32 dist:32, lo = ~z:6 ~pathCost:32 slot:26.  The comparison
  // ignores the slot bits.  dist and z are never negative.
  static constexpr int kLayerBits = 6;
  static constexpr int kSlotBits = 26;
  static_assert(kLayerBits + 32 + kSlotBits == 64);
  static constexpr uint64_t kSlotMask = (uint64_t(1) << kSlotBits) - 1;
  struct Key
  {
    uint64_t hi;
    uint64_t lo;
  };
  struct KeyLess  // "a has lower priority than b"
  {
    bool operator()(const Key& a, const Key& b) const
    {
      return a.hi != b.hi ? a.hi > b.hi
                          : (a.lo >> kSlotBits) > (b.lo >> kSlotBits);
    }
  };
  static Key makeKey(const FlexWavefrontGrid& g, uint32_t slot)
  {
    const uint64_t inv_z = ~uint64_t(g.z()) & ((uint64_t(1) << kLayerBits) - 1);
    const uint64_t inv_path_cost = uint32_t(~g.getPathCost());
    return {(uint64_t(g.getCost()) << 32) | uint32_t(g.getDist()),
            (inv_z << (32 + kSlotBits)) | (inv_path_cost << kSlotBits) | slot};
  }
  static uint32_t slotOf(const Key& key) { return key.lo & kSlotMask; }

 public:
  bool empty() const { return heap_.empty(); }
  const FlexWavefrontGrid& top() const { return slots_[slotOf(heap_.front())]; }
  void pop()
  {
    std::pop_heap(heap_.begin(), heap_.end(), KeyLess{});
    free_.push_back(slotOf(heap_.back()));
    heap_.pop_back();
  }
  void push(const FlexWavefrontGrid& in)
  {
    if (in.z() >= (1 << kLayerBits)) {
      throw std::length_error("FlexWavefront: too many routing layers");
    }
    uint32_t slot;
    if (!free_.empty()) {
      slot = free_.back();
      free_.pop_back();
      slots_[slot] = in;
    } else {
      if (slots_.size() > kSlotMask) {
        throw std::length_error("FlexWavefront: too many wavefront entries");
      }
      slot = slots_.size();
      slots_.push_back(in);
    }
    heap_.push_back(makeKey(in, slot));
    std::push_heap(heap_.begin(), heap_.end(), KeyLess{});
  }
  unsigned int size() const { return heap_.size(); }
  void cleanup()
  {
    heap_.clear();
    slots_.clear();
    free_.clear();
  }
  void fit()
  {
    cleanup();
    heap_.shrink_to_fit();
    slots_.shrink_to_fit();
    free_.shrink_to_fit();
  }

 private:
  std::vector<Key> heap_;
  std::vector<FlexWavefrontGrid> slots_;
  std::vector<uint32_t> free_;
};
}  // namespace drt
