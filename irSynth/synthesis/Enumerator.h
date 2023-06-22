#ifndef IRSYNTH_ENUMERATOR_H
#define IRSYNTH_ENUMERATOR_H

#include "execution/Executor.h"
#include "synthesis/CandidateStore.h"
#include "synthesis/Generators.h"
#include "synthesis/Stats.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>

struct SynthesisOptions {
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

struct SynthesisResult {
  CandidatePtr candidate;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
using SynthesisResultPtr = std::shared_ptr<SynthesisResult>;

void initializeCandidates(mlir::MLIRContext &ctx,
                          CandidateStorePtr &candidateStore,
                          mlir::Region::BlockArgListType functionArgs,
                          llvm::ArrayRef<int64_t> targetShape);

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);

SynthesisResultPtr
enumerateCandidates(mlir::MLIRContext &ctx, IExecutorPtr executor,
                    mlir::func::FuncOp inputFunction,
                    InitialCandidateGeneratorPtr initialCandidateGenerator,
                    CandidateStorePtr &candidateStore,
                    std::vector<mlir::RegisteredOperationName> &avaliableOps,
                    SynthesisOptions &options, SynthesisStats &stats);

#endif // IRSYNTH_ENUMERATOR_H
