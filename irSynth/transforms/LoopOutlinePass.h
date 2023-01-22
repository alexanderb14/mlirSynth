#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLoopOutlinePass();

struct LoopOutlinePass
    : public mlir::PassWrapper<LoopOutlinePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopOutlinePass)

  llvm::StringRef getArgument() const override { return "outline-loops"; }
  llvm::StringRef getDescription() const override {
    return "Outlines all top-level loops into seperate functions and calls "
           "them.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerLoopOutlinePass() { PassRegistration<LoopOutlinePass>(); }
} // namespace mlir
