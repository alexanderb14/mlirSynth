#ifndef IRSYNTH_SCOP_H
#define IRSYNTH_SCOP_H

#include "isl/isl_helper.h"
#include <unordered_map>

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

class ScopStmt {
public:
  ScopStmt(llvm::SmallVector<mlir::Operation *> allOps,
           llvm::SmallVector<mlir::Operation *> accessOps, std::string name)
      : allOps(allOps), accessOps(accessOps), name(name) {}

  isl::set domain() { return accessRelations[0].domain(); }

  void dump(llvm::raw_ostream &os, bool withLabels, bool withName,
            bool withDomain, bool withAccessOps);

public:
  llvm::SmallVector<mlir::Operation *> allOps;
  llvm::SmallVector<mlir::Operation *> accessOps;
  llvm::SmallVector<isl::map> accessRelations;
  std::string name;
};

class DependenceGraph {
public:
  class DependenceGraphNode;
  using DependenceGraphNodePtr = std::shared_ptr<DependenceGraphNode>;
  using DependenceGraphNodeWPtr = std::weak_ptr<DependenceGraphNode>;
  class DependenceGraphNode {
  public:
    DependenceGraphNode(ScopStmt *stmt) : stmt(stmt) {}

    ScopStmt *stmt;
    std::vector<DependenceGraphNodeWPtr> dependencies;
    std::vector<DependenceGraphNodeWPtr> dependents;
  };

  void computeDependencies();
  void dump(llvm::raw_ostream &os);

  llvm::SmallVector<DependenceGraphNodePtr> nodes;

private:
  int getNumDependencies();
};
using DependenceGraphPtr = std::shared_ptr<DependenceGraph>;

class Scop {
public:
  Scop(mlir::Operation *op);

  DependenceGraphPtr getDependenceGraph();

  ScopStmt *lookupStmtByName(std::string name);
  ScopStmt *lookupStmtByOp(mlir::Operation *op);
  llvm::SmallVector<ScopStmt> lookupStmtsByBlock(mlir::Block &block);

  void toDot(llvm::raw_ostream &os, Scop &scop);
  void toDotStmts(llvm::raw_ostream &os, Scop &scop);

  void dump(llvm::raw_ostream &os);

private:
  void buildScopStmts();
  void buildAccessRelationIslMaps();
  void computeFlowDependencies();

  isl::map getAccessRelationForOp(mlir::Operation *op, std::string &opName);

private:
  mlir::Operation *op;
  std::unordered_map<std::string, ScopStmt> namesToStmts;
  mlir::AsmState *asmState;
  isl::schedule schedule;
  isl::union_map flowDependencies;
  isl_ctx *ctx;
};

void dumpRelDetails(mlir::FlatAffineRelation rel);

namespace {
struct PolyhedralAnalysisPass
    : public mlir::PassWrapper<PolyhedralAnalysisPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolyhedralAnalysisPass)

  PolyhedralAnalysisPass() = default;
  PolyhedralAnalysisPass(const PolyhedralAnalysisPass &) {}

  Option<bool> dump{*this, "dump", llvm::cl::desc("Dump all info"),
                    llvm::cl::init(false)};

  Option<bool> dot{*this, "dot", llvm::cl::desc("Dot graph"),
                   llvm::cl::init(false)};

  Option<bool> dotStmts{*this, "dot-stmts",
                        llvm::cl::desc("Dot graph of statement contents"),
                        llvm::cl::init(false)};

  Option<bool> dumpDependenceGraph{*this, "dump-dependence-graph",
                                   llvm::cl::desc("Dump dependence graph"),
                                   llvm::cl::init(false)};

  llvm::StringRef getArgument() const override { return "polyhedral-analysis"; }
  llvm::StringRef getDescription() const override {
    return "Polyhedral dependence analysis using ISL.";
  }
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    Scop scop(op);

    if (dump) {
      scop.dump(llvm::outs());
      llvm::outs() << "\n";
    }

    if (dot) {
      scop.toDot(llvm::outs(), scop);
      llvm::outs() << "\n";
    }

    if (dotStmts) {
      scop.toDotStmts(llvm::outs(), scop);
      llvm::outs() << "\n";
    }

    if (dumpDependenceGraph) {
      auto dg = scop.getDependenceGraph();
      dg->dump(llvm::outs());
      llvm::outs() << "\n";
    }
  }
};
} // namespace

namespace mlir {
inline void registerPolyhedralAnalysisPass() { PassRegistration<PolyhedralAnalysisPass>(); }
} // namespace mlir

#endif // IRSYNTH_SCOP_H
