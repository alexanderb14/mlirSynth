#include "Guide.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>
#include <unordered_map>

#include "analysis/PolyhedralAnalysis.h"

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

std::vector<std::string> predictOps(std::vector<std::string> &supportedOps,
                                    mlir::Operation *op) {
  Scop scop(op);
  auto dg = scop.getDependenceGraph();
  auto g = constructBoostGraph(dg);

  // Make decisions.
  std::vector<std::string> ops;
  if (countNumArithDiv(op) > 0)
    ops.emplace_back("chlo.broadcast_divide");
  if (countNumArithAdd(op) > 0)
    ops.emplace_back("chlo.broadcast_add");
  if (countNumArithSub(op) > 0)
    ops.emplace_back("chlo.broadcast_subtract");
  if (countNumArithMul(op) > 0)
    ops.emplace_back("chlo.broadcast_multiply");

  if (computeNumCyclesWithSelfEdges(g) > 0) {
    ops.emplace_back("mhlo.dot");
    ops.emplace_back("mhlo.dot_general");
    ops.emplace_back("mhlo.reduce");
  }

  // If we didn't match any ops, add all of them.
  if (ops.empty())
    return supportedOps;

  return ops;
}
