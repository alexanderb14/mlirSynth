#include "Guide.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>
#include <unordered_map>

#include "analysis/PolyhedralAnalysis.h"

namespace boost {
inline void throw_exception(std::exception const &e) {
  llvm::errs() << "Boost exception: " << e.what() << "\n";
}
} // namespace boost

using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int>>;

namespace boost {
void renumber_vertex_indices(Graph const &) {}
} // namespace boost

Graph constructBoostGraph(DependenceGraphPtr& dg) {
  Graph g;
  std::unordered_map<ScopStmt *, Graph::vertex_descriptor> vertexMap;

  // Add nodes.
  for (auto &node : dg->nodes) {
    vertexMap[node->stmt] = boost::add_vertex(g);
  }

  // Add edges.
  for (auto &node : dg->nodes) {
    for (auto &dep : node->dependents) {
      auto *src = node->stmt;
      auto *dst = dep.lock()->stmt;

      auto srcVertex = vertexMap[src];
      auto dstVertex = vertexMap[dst];

      boost::add_edge(srcVertex, dstVertex, g);
    }
  }

  return g;
}

struct CycleRecorder {
  template <typename Path, typename Graph>
  void cycle(const Path &p, const Graph &g) {
    numCycles++;
  }
  int numCycles = 0;
};

int computeNumCycles(Graph const &g) {
  int numCycles = 0;
  // Check for self edges
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      if (boost::target(e, g) == v) {
        numCycles++;
      }
    }
  }

  // Check for cycles
  CycleRecorder cycleRecorder;
  boost::tiernan_all_cycles(g, cycleRecorder);
  numCycles += cycleRecorder.numCycles;

  return numCycles;
}

int countNumArithDiv(mlir::Operation *op) {
  int numArithDiv = 0;
  op->walk([&](mlir::arith::DivFOp op) {
    numArithDiv++;
  });
  return numArithDiv;
}

int countNumArithAdd(mlir::Operation *op) {
  int numArithAdd = 0;
  op->walk([&](mlir::arith::AddFOp op) {
    numArithAdd++;
  });
  return numArithAdd;
}

int countNumArithSub(mlir::Operation *op) {
  int numArithSub = 0;
  op->walk([&](mlir::arith::SubFOp op) {
    numArithSub++;
  });
  return numArithSub;
}

int countNumArithMul(mlir::Operation *op) {
  int numArithMul = 0;
  op->walk([&](mlir::arith::MulFOp op) {
    numArithMul++;
  });
  return numArithMul;
}

std::vector<std::string> predictOps(mlir::Operation *op) {
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

  if(computeNumCycles(g) > 0) {
    ops.emplace_back("mhlo.dot");
    ops.emplace_back("mhlo.dot_general");
    ops.emplace_back("mhlo.reduce");
  }
  return ops;
}
