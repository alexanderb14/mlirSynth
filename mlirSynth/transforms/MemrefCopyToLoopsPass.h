#ifndef IRSYNTH_TRANSFORMS_MEMREFCOPYTOLOOPSPASS_H
#define IRSYNTH_TRANSFORMS_MEMREFCOPYTOLOOPSPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMemrefCopyToLoopsPass();

struct MemrefCopyToLoopsPass
    : public mlir::PassWrapper<MemrefCopyToLoopsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemrefCopyToLoopsPass)

  llvm::StringRef getArgument() const override { return "memref-copy-to-loops"; }
  llvm::StringRef getDescription() const override {
    return "Convert memref.copy ops to loops.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerMemrefCopyToLoopsPass() { PassRegistration<MemrefCopyToLoopsPass>(); }
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_MEMREFCOPYTOLOOPSPASS_H
