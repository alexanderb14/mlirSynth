#ifndef IRSYNTH_TRANSFORMS_TARGETOUTLINEPASS_H
#define IRSYNTH_TRANSFORMS_TARGETOUTLINEPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createTargetOutlinePass();

struct TargetOutlinePass
    : public mlir::PassWrapper<TargetOutlinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TargetOutlinePass)

  llvm::StringRef getArgument() const override { return "outline-target"; }
  llvm::StringRef getDescription() const override {
    return "Outlines the target program into a seperate function and calls "
           "it.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerTargetOutlinePass() { PassRegistration<TargetOutlinePass>(); }
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_TARGETOUTLINEPASS_H
