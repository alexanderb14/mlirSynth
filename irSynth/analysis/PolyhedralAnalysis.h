#ifndef IRSYNTH_SCOP_H
#define IRSYNTH_SCOP_H

#include "isl/isl_helper.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"

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

class Scop {
public:
  Scop(mlir::Operation *op);

  ScopStmt *lookupStmt(mlir::Operation *op);
  llvm::SmallVector<ScopStmt> lookupStmts(mlir::Block &block);

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
  llvm::SmallVector<ScopStmt> stmts;
  mlir::AsmState *asmState;
  isl::schedule schedule;
  isl::union_map flowDependencies;
  isl_ctx *ctx;
};

void dumpRelDetails(mlir::FlatAffineRelation rel);

#endif // IRSYNTH_SCOP_H
