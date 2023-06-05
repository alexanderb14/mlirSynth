#ifndef IRSYNTH_ENUMERATOR_H
#define IRSYNTH_ENUMERATOR_H

#include "enumeration/CandidateStore.h"
#include "enumeration/Generators.h"
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
  int timeoutPerFunction;
  bool ignoreEquivalentCandidates;
  bool ignoreTypes;
  bool skipTypeInference;
  bool withCopyArgs;
  bool guide;
  bool distribute;
};

struct EnumerationResult {
  CandidatePtr candidate;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
using EnumerationResultPtr = std::shared_ptr<EnumerationResult>;

void initializeCandidates(mlir::MLIRContext &ctx,
                          CandidateStorePtr &candidateStore,
                          mlir::Region::BlockArgListType functionArgs,
                          llvm::ArrayRef<int64_t> targetShape);

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);

EnumerationResultPtr
enumerateCandidates(mlir::MLIRContext &ctx, IExecutorPtr executor,
                    mlir::func::FuncOp inputFunction,
                    InitialCandidateGeneratorPtr initialCandidateGenerator,
                    CandidateStorePtr &candidateStore,
                    std::vector<mlir::RegisteredOperationName> &avaliableOps,
                    EnumerationOptions &options, EnumerationStats &stats);

#endif // IRSYNTH_ENUMERATOR_H
