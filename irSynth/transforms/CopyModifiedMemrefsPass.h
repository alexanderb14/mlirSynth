#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCopyModifiedMemrefsPass();

struct CopyModifiedMemrefsPass
    : public mlir::PassWrapper<CopyModifiedMemrefsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CopyModifiedMemrefsPass)

  llvm::StringRef getArgument() const override {
    return "copy-modified-memrefs";
  }
  llvm::StringRef getDescription() const override {
    return "Creates copies of stored memrefs and replaces their original uses "
           "with them.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerCopyModifiedMemrefsPass() {
  PassRegistration<CopyModifiedMemrefsPass>();
}
} // namespace mlir
