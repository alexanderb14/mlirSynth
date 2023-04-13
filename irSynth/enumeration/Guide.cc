#include "Guide.h"

#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/ADT/ArrayRef.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include <unordered_map>

using namespace mlir;

template <typename T>
int countNumOps(Operation *op) {
  int numOps = 0;
  op->walk([&](T op) { numOps++; });
  return numOps;
}

int computeNumCyclesWithSelfEdges(BoostGraph &g) {
  // Check for cycles
  int numCycles = computeNumCycles(g);

  // Check for self edges
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      if (boost::target(e, g) == v) {
        numCycles++;
      }
    }
  }

  return numCycles;
}

llvm::SmallVector<llvm::ArrayRef<int64_t>>
getUsedMemrefShapes(Operation *op) {
  llvm::SmallVector<llvm::ArrayRef<int64_t>> memrefShapes;

  // Collect shapes
  for (auto operand : op->getOperands()) {
    // Case 1: Operand is an argument
    if (auto lhsArg = operand.dyn_cast<BlockArgument>()) {
      if (auto memrefType = lhsArg.getType().dyn_cast<MemRefType>()) {
        auto memrefShape = memrefType.getShape();
        memrefShapes.push_back(memrefShape);
      }
    }

    // Case 2: Operand is an affine.load
    if (auto *lhsOp = operand.getDefiningOp()) {
      if (auto loadOp = dyn_cast<AffineLoadOp>(lhsOp)) {
        auto memrefType = loadOp.getMemRefType();
        auto memrefShape = memrefType.getShape();
        memrefShapes.push_back(memrefShape);
      }
    }
  }

  // Recursive walk the use chain
  for (auto operand : op->getOperands()) {
    if (auto *definingOp = operand.getDefiningOp()) {
      auto memrefShapesOperand = getUsedMemrefShapes(definingOp);
      memrefShapes.append(memrefShapesOperand.begin(),
                          memrefShapesOperand.end());
    }
  }

  return memrefShapes;
}

int getMaxArgDim(Operation *op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  auto funcType = funcOp.getFunctionType();
  auto funcArgs = funcType.getInputs();

  // Get the maximum dimension of the arguments
  int maxArgDim = 0;
  for (auto arg : funcArgs) {
    if (auto memrefType = arg.dyn_cast<MemRefType>()) {
      auto memrefShape = memrefType.getShape();
      maxArgDim = std::max(maxArgDim, (int)memrefShape.size());
    }
  }

  return maxArgDim;
}

int countNumMultipliedMismatchingMemrefAccesses(Operation *op) {
  int numMultipliedMismatchingMemrefs = 0;

  // Walk over affine.store operations
  op->walk([&](arith::MulFOp mulOp) {
    auto memrefShapes = getUsedMemrefShapes(mulOp);

    // Check if all memref shapes have a matching dimension with at least one
    // other memref shape. Matching dimensions can be their first-last or
    // last-first dimensions.
    for (auto memrefShape : memrefShapes) {
      bool hasMatchingDimension = false;
      for (auto memrefShapeOther : memrefShapes) {
        if (memrefShape.size() != memrefShapeOther.size())
          continue;
        if (memrefShape.empty() || memrefShapeOther.empty())
          continue;

        if (memrefShape[0] == memrefShapeOther.back() ||
            memrefShape.back() == memrefShapeOther[0]) {
          hasMatchingDimension = true;
          break;
        }
      }
      if (!hasMatchingDimension) {
        numMultipliedMismatchingMemrefs++;
      }
    }
  });

  return numMultipliedMismatchingMemrefs;
}

int countNumLoopBoundMaps(Operation *op) {
  int numLoopBoundMaps = 0;
  op->walk([&](AffineForOp forOp) {
    if (!forOp.hasConstantUpperBound()) {
      numLoopBoundMaps++;
    }
  });
  return numLoopBoundMaps;
}

std::vector<std::string> predictOps(std::vector<std::string> &supportedOps,
                                    Operation *op) {
  Scop scop(op);
  auto dg = scop.getDependenceGraph();
  auto g = constructBoostGraph(dg);

  // Element wise heuristics
  std::vector<std::string> ops;
  if (countNumOps<arith::DivFOp>(op) > 0)
    ops.emplace_back("chlo.broadcast_divide");
  if (countNumOps<arith::AddFOp>(op) > 0)
    ops.emplace_back("chlo.broadcast_add");
  if (countNumOps<arith::SubFOp>(op) > 0)
    ops.emplace_back("chlo.broadcast_subtract");
  if (countNumOps<arith::MulFOp>(op) > 0)
    ops.emplace_back("chlo.broadcast_multiply");

  if (countNumOps<math::SqrtOp>(op) > 0)
    ops.emplace_back("stablehlo.sqrt");
  if (countNumOps<arith::SelectOp>(op) > 0) {
    ops.emplace_back("stablehlo.select");
  }
  if (countNumOps<arith::CmpFOp>(op) > 0) {
    ops.emplace_back("stablehlo.compare");
  }

  // Transpose heuristics
  if (countNumMultipliedMismatchingMemrefAccesses(op) > 0)
    ops.emplace_back("stablehlo.transpose");

  // Select heuristics
  if (countNumLoopBoundMaps(op) > 0)
    ops.emplace_back("stablehlo.select");

  // Reduction heuristics
  if (computeNumCyclesWithSelfEdges(g) > 0) {
    if (getMaxArgDim(op) > 2) {
      ops.emplace_back("stablehlo.dot_general");
    } else {
      ops.emplace_back("stablehlo.dot");
    }

    ops.emplace_back("stablehlo.reduce");
    ops.emplace_back("linalg.matmul");
    ops.emplace_back("linalg.matvec");
  }

  // If we didn't match any ops, add all of them.
  if (ops.empty())
    return supportedOps;

  return ops;
}
