#ifndef IRSYNTH_ENUMERATOR_H
#define IRSYNTH_ENUMERATOR_H

#include "enumeration/Candidate.h"
#include "enumeration/Stats.h"
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
  bool printArgsAndResults;
  int maxNumOps;
  bool ignoreEquivalentCandidates;
};

void initializeCandidates(mlir::MLIRContext &ctx,
                          CandidateStorePtr &candidateStore);

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);

using ModuleAndArgIds =
    std::tuple<mlir::OwningOpRef<mlir::ModuleOp>, std::vector<unsigned>>;
ModuleAndArgIds
enumerateCandidates(mlir::MLIRContext &ctx, IExecutorPtr executor,
                    mlir::func::FuncOp inputFunction,
                    CandidateStorePtr &candidateStore,
                    std::vector<mlir::RegisteredOperationName> &avaliableOps,
                    EnumerationOptions &options);

#endif // IRSYNTH_ENUMERATOR_H
