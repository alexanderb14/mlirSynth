#ifndef IRSYNTH_TRANSFORMS_PREPARETARGETPASS_H
#define IRSYNTH_TRANSFORMS_PREPARETARGETPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPrepareTargetPass();

struct PrepareTargetPass
    : public mlir::PassWrapper<PrepareTargetPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareTargetPass)

  llvm::StringRef getArgument() const override { return "prepare-target"; }
  llvm::StringRef getDescription() const override {
    return "Prepares the function that has an irsynth.target argument for compilation with XLA.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerPrepareTargetPass() { PassRegistration<PrepareTargetPass>(); }
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_PREPARETARGETPASS_H
