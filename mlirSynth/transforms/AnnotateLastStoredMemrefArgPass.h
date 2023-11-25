#ifndef IRSYNTH_TRANSFORMS_ANNOTATELASTSTOREDMEMREFARGPASS_H
#define IRSYNTH_TRANSFORMS_ANNOTATELASTSTOREDMEMREFARGPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAnnotateLastStoredMemrefArgPass();

struct AnnotateLastStoredMemrefArgPass
    : public mlir::PassWrapper<AnnotateLastStoredMemrefArgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateLastStoredMemrefArgPass)

  llvm::StringRef getArgument() const override { return "annotate-output-arg"; }
  llvm::StringRef getDescription() const override {
    return "Annotates the output argument.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerAnnotateLastStoredMemrefArgPass() {
  PassRegistration<AnnotateLastStoredMemrefArgPass>();
}
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_ANNOTATELASTSTOREDMEMREFARGPASS_H
