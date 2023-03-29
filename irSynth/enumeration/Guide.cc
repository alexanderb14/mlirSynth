#include "Guide.h"

#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/ArrayRef.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include <unordered_map>

using namespace mlir;

int countNumArithDiv(mlir::Operation *op) {
  int numArithDiv = 0;
  op->walk([&](mlir::arith::DivFOp op) { numArithDiv++; });
  return numArithDiv;
}

int countNumArithAdd(mlir::Operation *op) {
  int numArithAdd = 0;
  op->walk([&](mlir::arith::AddFOp op) { numArithAdd++; });
  return numArithAdd;
}

int countNumArithSub(mlir::Operation *op) {
  int numArithSub = 0;
  op->walk([&](mlir::arith::SubFOp op) { numArithSub++; });
  return numArithSub;
}

int countNumArithMul(mlir::Operation *op) {
  int numArithMul = 0;
  op->walk([&](mlir::arith::MulFOp op) { numArithMul++; });
  return numArithMul;
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
getUsedMemrefShapes(mlir::Operation *op) {
  llvm::SmallVector<llvm::ArrayRef<int64_t>> memrefShapes;

  // Collect shapes
  for (auto operand : op->getOperands()) {
    // Case 1: Operand is an argument
    if (auto lhsArg = operand.dyn_cast<mlir::BlockArgument>()) {
      if (auto memrefType = lhsArg.getType().dyn_cast<mlir::MemRefType>()) {
        auto memrefShape = memrefType.getShape();
        memrefShapes.push_back(memrefShape);
      }
    }

    // Case 2: Operand is an affine.load
    if (auto *lhsOp = operand.getDefiningOp()) {
      if (auto loadOp = mlir::dyn_cast<mlir::AffineLoadOp>(lhsOp)) {
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

int countNumMultipliedMismatchingMemrefAccesses(mlir::Operation *op) {
  int numMultipliedMismatchingMemrefs = 0;

  // Walk over affine.store operations
  op->walk([&](mlir::arith::MulFOp mulOp) {
    auto memrefShapes = getUsedMemrefShapes(mulOp);

    // Check if all memref shapes have a matching dimension with at least one
    // other memref shape. Matching dimensions can be their first-last or
    // last-first dimensions.
    for (auto memrefShape : memrefShapes) {
      bool hasMatchingDimension = false;
      for (auto memrefShapeOther : memrefShapes) {
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

int countNumLoopBoundMaps(mlir::Operation *op) {
  int numLoopBoundMaps = 0;
  op->walk([&](mlir::AffineForOp forOp) {
    if (forOp.getUpperBoundMap() || forOp.getLowerBoundMap()) {
      numLoopBoundMaps++;
    }
  });
  return numLoopBoundMaps;
}

std::vector<std::string> predictOps(std::vector<std::string> &supportedOps,
                                    mlir::Operation *op) {
  Scop scop(op);
  auto dg = scop.getDependenceGraph();
  auto g = constructBoostGraph(dg);

  // Element wise heuristics
  std::vector<std::string> ops;
  if (countNumArithDiv(op) > 0)
    ops.emplace_back("chlo.broadcast_divide");
  if (countNumArithAdd(op) > 0)
    ops.emplace_back("chlo.broadcast_add");
  if (countNumArithSub(op) > 0)
    ops.emplace_back("chlo.broadcast_subtract");
  if (countNumArithMul(op) > 0)
    ops.emplace_back("chlo.broadcast_multiply");

  // Transpose heuristics
  if (countNumMultipliedMismatchingMemrefAccesses(op) > 0) {
    ops.emplace_back("mhlo.transpose");
  }

  // Select heuristics
  if (countNumLoopBoundMaps(op) > 0) {
    ops.emplace_back("mhlo.select");
  }

  // Reduction heuristics
  if (computeNumCyclesWithSelfEdges(g) > 0) {
    ops.emplace_back("mhlo.dot_general");
    ops.emplace_back("mhlo.reduce");
  }

  // If we didn't match any ops, add all of them.
  if (ops.empty())
    return supportedOps;

  return ops;
}
