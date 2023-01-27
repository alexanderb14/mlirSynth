#ifndef IRSYNTH_TRANSFORMS_LOOPDISTRIBUTIONPASS_H
#define IRSYNTH_TRANSFORMS_LOOPDISTRIBUTIONPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLoopDistributionPass();

struct LoopDistributionPass
    : public mlir::PassWrapper<LoopDistributionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopDistributionPass)

  llvm::StringRef getArgument() const override { return "distribute-loops"; }
  llvm::StringRef getDescription() const override {
    return "Distributes loops.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerLoopDistributionPass() {
  PassRegistration<LoopDistributionPass>();
}
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_LOOPDISTRIBUTIONPASS_H
