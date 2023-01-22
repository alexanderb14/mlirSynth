#include "LoopDistributionPass.h"

#include "Utils.h"
#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "boost/graph/topological_sort.hpp"

using namespace mlir;

void eliminateSelfDependencies(BoostGraph &g) {
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      if (boost::target(e, g) == v) {
        boost::remove_edge(e, g);
      }
    }
  }
}

void isolateStmt(ScopStmt *stmt) {
  // Isolate the statements operations, while keeping them in their loop nest.
  // This means that if the statement is in a nested loop, we need to create
  // a new loop for the statement, and move the statement into that loop.
  // Finally, we need to move the statement out of the old loop nest.
  auto *firstStmtOp = stmt->allOps[0];

  std::vector<mlir::Operation *> loopStack;
  auto *parent = firstStmtOp->getParentOp();
  while (isa<AffineForOp>(parent)) {
    loopStack.push_back(parent);
    parent = parent->getParentOp();
  }

  auto *outestForOp = loopStack.back();

  // Create a builder for the new loop.
  OpBuilder builder(outestForOp->getContext());
  builder.setInsertionPoint(outestForOp);

  // For each loop in the loop stack, create a new loop that looks like the old
  // one.
  BlockAndValueMapping mapping;
  auto newLoop = dyn_cast<AffineForOp>(outestForOp);
  for (auto *loop : loopStack) {
    auto forOp = dyn_cast<AffineForOp>(loop);
    newLoop = builder.create<AffineForOp>(
        forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
        forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(),
        forOp.getStep());
    builder.setInsertionPointToStart(newLoop.getBody());
    mapping.map(forOp.getInductionVar(), newLoop.getInductionVar());
  }

  // Move the statements operations into the new loop.
  for (auto *op : stmt->allOps) {
    op->moveBefore(newLoop.getLoopBody().back().getTerminator());

    // Remap the operands of the operation.
    for (auto &operand : op->getOpOperands()) {
      operand.set(mapping.lookupOrDefault(operand.get()));
    }
  }
}

void distributeLoops(Operation *op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  auto originalLoops = getTopLevelLoops(funcOp);

  Scop scop(op);
  auto dg = scop.getDependenceGraph();

  // If there are cyclic dependencies, we cannot distribute the loops, therefore
  // bail out and don't apply this transformation.
  // TODO: We could still apply this transformation if cyclic dependencies are
  // present. We need to detect the cycles and merge the nodes in the cycle into
  // a single node, then apply topological sorting.
  auto g = constructBoostGraph(dg);
  int numCycles = computeNumCycles(g);
  if (numCycles > 0)
    return;

  // Get a topological order of the graph.
  // - First, make the graph a DAG by removing self dependencies.
  eliminateSelfDependencies(g);
  // - Then, compute the topological order.
  std::vector<int> topoOrder;
  boost::topological_sort(g, std::back_inserter(topoOrder));
  std::reverse(topoOrder.begin(), topoOrder.end());

  // Distribute the loops according to the topological order.
  for (auto idx : topoOrder) {
    auto *stmt = g[idx].stmt;
    isolateStmt(stmt);
  }

  // Remove the original loops, which are empty by now.
  for (auto *loop : originalLoops) {
    loop->walk([](Operation *op) {
      llvm::outs() << op->getName().getStringRef() << "\n";
      op->erase();
    });
  }
}

void LoopDistributionPass::runOnOperation() {
  auto operation = getOperation();
  for (auto func : operation.getOps<func::FuncOp>()) {
    distributeLoops(func);
  }
  operation->dump();
}

std::unique_ptr<OperationPass<ModuleOp>> createLoopDistributionPass() {
  return std::make_unique<LoopDistributionPass>();
}
