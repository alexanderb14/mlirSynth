#ifndef TOOLS_SYNTHESIZER_ENUMERATOR_H
#define TOOLS_SYNTHESIZER_ENUMERATOR_H

#include "Candidate.h"
#include "Stats.h"

#include "execution/Executor.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>

struct EnumerationOptions {
  bool printStatusNames;
  bool printStatusTiles;
  bool printValidCandidates;
  bool printInvalidCandidates;
  bool printStats;
  int maxNumOps;
  bool ignoreEquivalentCandidates;
};

void initializeCandidates(mlir::MLIRContext &ctx,
                          CandidateStorePtr &candidateStore);

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);

bool enumerateCandidates(
    mlir::MLIRContext &ctx, IExecutorPtr executor,
    mlir::func::FuncOp inputFunction, CandidateStorePtr &candidateStore,
    std::vector<mlir::RegisteredOperationName> &avaliableOps,
    EnumerationOptions &options);

#endif // TOOLS_SYNTHESIZER_ENUMERATOR_H
