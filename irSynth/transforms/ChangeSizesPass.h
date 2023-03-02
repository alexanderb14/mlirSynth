#ifndef IRSYNTH_TRANSFORMS_CHANGESIZEPASS_H
#define IRSYNTH_TRANSFORMS_CHANGESIZEPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createChangeSizesPass();

struct ChangeSizesPass
    : public mlir::PassWrapper<ChangeSizesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ChangeSizesPass)

  ChangeSizesPass() = default;
  ChangeSizesPass(const ChangeSizesPass& pass)  : PassWrapper(pass) {}

  llvm::StringRef getArgument() const override { return "change-sizes"; }
  llvm::StringRef getDescription() const override {
    return "Minifies memrefs according to a prime number sequence.";
  }
  void runOnOperation() override;

  Option<std::string> mode{*this, "mode", llvm::cl::desc("Minify or Restore"),
                                    llvm::cl::init("minify")};
};

namespace mlir {
inline void registerChangeSizesPass() { PassRegistration<ChangeSizesPass>(); }
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_CHANGESIZEPASS_H
