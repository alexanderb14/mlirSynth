#ifndef IRSYNTH_TRANSFORMS_FOLDTOTENSORTOMENREFPAIRPASS_H
#define IRSYNTH_TRANSFORMS_FOLDTOTENSORTOMENREFPAIRPASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createFoldToTensorToMemrefPairPass();

struct FoldToTensorToMemrefPairPass
    : public mlir::PassWrapper<FoldToTensorToMemrefPairPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FoldToTensorToMemrefPairPass)

  llvm::StringRef getArgument() const override {
    return "fold-totensor-tomemref-pairs";
  }
  llvm::StringRef getDescription() const override {
    return "Folds to_tensor and to_memref pairs.";
  }
  void runOnOperation() override;
};

namespace mlir {
inline void registerFoldToTensorToMemrefPairPass() {
  PassRegistration<FoldToTensorToMemrefPairPass>();
}
} // namespace mlir

#endif // IRSYNTH_TRANSFORMS_FOLDTOTENSORTOMENREFPAIRPASS_H
