#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMemrefMinifyPass();

struct MemrefMinifyPass
    : public mlir::PassWrapper<MemrefMinifyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemrefMinifyPass)

  llvm::StringRef getArgument() const override { return "minify-memrefs"; }
  llvm::StringRef getDescription() const override {
    return "Minifies memrefs according to a prime number sequence.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerMemrefMinifyPass() { PassRegistration<MemrefMinifyPass>(); }
} // namespace mlir
