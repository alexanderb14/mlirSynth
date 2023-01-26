#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createCleanupPass();

struct CleanupPass
    : public mlir::PassWrapper<CleanupPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CleanupPass)

  llvm::StringRef getArgument() const override { return "cleanup"; }
  llvm::StringRef getDescription() const override {
    return "Cleans up the IR after the synthesis.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerCleanupPass() { PassRegistration<CleanupPass>(); }
} // namespace mlir
