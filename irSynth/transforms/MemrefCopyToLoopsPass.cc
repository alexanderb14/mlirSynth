#include "MemrefCopyToLoopsPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

void MemrefCopyToLoopsPass::runOnOperation() {
  auto operation = getOperation();
  auto *ctx = operation->getContext();

  // Find all memref.copy ops.
  SmallVector<memref::CopyOp, 4> copyOps;
  operation->walk([&](memref::CopyOp copyOp) { copyOps.push_back(copyOp); });

  // Replace each memref.copy op with a loop nest.
  for (auto copyOp : copyOps) {
    // Get the source and destination memrefs.
    auto sourceMemref = copyOp.getSource();
    auto destMemref = copyOp.getTarget();

    // Get the source and destination memref types.
    auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto destMemrefType = destMemref.getType().cast<MemRefType>();

    // Get the source and destination memref shapes.
    auto sourceShape = sourceMemrefType.getShape();
    auto destShape = destMemrefType.getShape();

    // Create a loop for each dimension and create a memref.load from the source memref and a memref.store to the destination memref.
    auto builder = OpBuilder(copyOp);
    auto loc = copyOp.getLoc();


    auto beforeCopy = builder.saveInsertionPoint();

    SmallVector<mlir::Value> indices;
    for (auto dimSize : sourceShape) {
      // Create for op.
      auto forOp = builder.create<scf::ForOp>(
          UnknownLoc::get(ctx),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

      indices.push_back(forOp.getInductionVar());

      // Set insertion point inside body for next loop to be inserted.
      builder.setInsertionPointToStart(forOp.getBody());

      // If we are in the innermost loop, create the declaration.
      if (indices.size() == sourceShape.size()) {
        // Create load.
        mlir::Type lhsType =
            sourceMemrefType.getElementType();
        auto lhsLoad = builder.create<memref::LoadOp>(
            UnknownLoc::get(ctx), lhsType, sourceMemref, indices);

        // Create store.
        builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                        lhsLoad, destMemref, indices);
      }
    }

    builder.restoreInsertionPoint(beforeCopy);

    // Erase the memref.copy op.
    copyOp.erase();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createMemrefCopyToLoopsPass() {
  return std::make_unique<MemrefCopyToLoopsPass>();
}

