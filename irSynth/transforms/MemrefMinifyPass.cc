#include "MemrefMinifyPass.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

int nextPrime(int n) {
  if (n <= 2)
    return 2;
  if (n <= 3)
    return 3;
  if (n % 2 == 0 || n % 3 == 0)
    return nextPrime(n + 1);
  for (int i = 5; i * i <= n; i = i + 6)
    if (n % i == 0 || n % (i + 2) == 0)
      return nextPrime(n + 1);
  return n;
}

llvm::DenseMap<int64_t, int64_t> getMinifedDimensionMap(ModuleOp &op) {
  bool debug = false;

  // Collect all memref types.
  llvm::SetVector<MemRefType> memrefTypes;
  op->walk([&](Operation *op) {
    // - Collect from operation operands.
    for (auto operand : op->getOperands())
      if (operand.getType().isa<MemRefType>())
        memrefTypes.insert(operand.getType().cast<MemRefType>());
    // - Collect from operation results.
    for (auto type : op->getResultTypes())
      if (type.isa<MemRefType>())
        memrefTypes.insert(type.cast<MemRefType>());
  });

  if (debug) {
    llvm::outs() << "Collected memref types:\n";
    for (auto type : memrefTypes)
      llvm::errs() << type << "\n";
  }

  // Collect all dimensions.
  llvm::SetVector<int64_t> dimensions;
  for (auto type : memrefTypes) {
    for (auto dim : type.getShape())
      dimensions.insert(dim);
  }

  if (debug) {
    llvm::outs() << "Collected dimensions:\n";
    for (auto dim : dimensions)
      llvm::errs() << dim << "\n";
  }

  // Sort them.
  llvm::SmallVector<int64_t> sortedDimensions;
  for (auto dim : dimensions)
    sortedDimensions.push_back(dim);
  std::sort(sortedDimensions.begin(), sortedDimensions.end());

  if (debug) {
    llvm::outs() << "Sorted dimensions:\n";
    for (auto dim : sortedDimensions)
      llvm::errs() << dim << "\n";
  }

  // Create a mapping from sorted dimensions to their minified values. Minified
  // values are prime numbers.
  llvm::DenseMap<int64_t, int64_t> minifiedDimensions;
  int64_t minifiedValue = 3;
  for (auto dim : sortedDimensions) {
    minifiedDimensions[dim] = minifiedValue;
    minifiedValue = nextPrime(minifiedValue + 1);
  }

  if (debug) {
    llvm::outs() << "Minified dimensions:\n";
    for (auto dim : dimensions)
      llvm::errs() << dim << " -> " << minifiedDimensions[dim] << "\n";
  }

  return minifiedDimensions;
}

void minifyMemrefs(ModuleOp &op,
                   llvm::DenseMap<int64_t, int64_t> &minifiedDimensions) {
  // In function signatures.
  for (auto func : op.getOps<func::FuncOp>()) {
    auto type = func.getFunctionType();
    // - Minify memref types in function arguments.
    llvm::SmallVector<Type> minifiedArgTypes;
    for (auto argType : type.getInputs()) {
      if (argType.isa<MemRefType>()) {
        auto memrefType = argType.cast<MemRefType>();
        llvm::SmallVector<int64_t> minifiedShape;
        for (auto dim : memrefType.getShape()) {
          long newDim;
          if (minifiedDimensions.count(dim) == 0)
            newDim = dim;
          else
            newDim = minifiedDimensions[dim];
          minifiedShape.push_back(newDim);
        }
        auto minifiedType =
            MemRefType::get(minifiedShape, memrefType.getElementType());
        minifiedArgTypes.push_back(minifiedType);
      } else {
        minifiedArgTypes.push_back(argType);
      }
    }

    // - Minify memref types in function results.
    llvm::SmallVector<Type> minifiedResultTypes;
    for (auto resultType : type.getResults()) {
      if (resultType.isa<MemRefType>()) {
        auto memrefType = resultType.cast<MemRefType>();
        llvm::SmallVector<int64_t> minifiedShape;
        for (auto dim : memrefType.getShape()) {
          long newDim;
          if (minifiedDimensions.count(dim) == 0)
            newDim = dim;
          else
            newDim = minifiedDimensions[dim];
          minifiedShape.push_back(newDim);
        }
        auto minifiedType =
            MemRefType::get(minifiedShape, memrefType.getElementType());
        minifiedResultTypes.push_back(minifiedType);
      } else {
        minifiedResultTypes.push_back(resultType);
      }
    }

    // - Set the new function type.
    auto minifiedType = FunctionType::get(type.getContext(), minifiedArgTypes,
                                          minifiedResultTypes);
    func.setType(minifiedType);
  }

  // In operations.
  op->walk([&](Operation *op) {
    // - Minify memref types in operation operands.
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<MemRefType>()) {
        auto type = operand.getType().cast<MemRefType>();
        llvm::SmallVector<int64_t> minifiedShape;
        for (auto dim : type.getShape()) {
          long newDim;
          if (minifiedDimensions.count(dim) == 0)
            newDim = dim;
          else
            newDim = minifiedDimensions[dim];
          minifiedShape.push_back(newDim);
        }
        auto minifiedType =
            MemRefType::get(minifiedShape, type.getElementType());
        operand.setType(minifiedType);
      }
    }
    // - Minify memref types in operation results.
    for (auto type : op->getResultTypes()) {
      if (type.isa<MemRefType>()) {
        auto resType = type.cast<MemRefType>();
        llvm::SmallVector<int64_t> minifiedShape;
        for (auto dim : resType.getShape()) {
          long newDim;
          if (minifiedDimensions.count(dim) == 0)
            newDim = dim;
          else
            newDim = minifiedDimensions[dim];
          minifiedShape.push_back(newDim);
        }
        auto minifiedType =
            MemRefType::get(minifiedShape, resType.getElementType());
        resType = minifiedType;
      }
    }
  });
}

void minifyLoopBounds(ModuleOp &op,
                      llvm::DenseMap<int64_t, int64_t> &minifiedDimensions) {
  bool debug = false;

  op->walk([&](Operation *op) {
    if (isa<AffineForOp>(op)) {
      auto forOp = cast<AffineForOp>(op);
      auto ubMap = forOp.getUpperBoundMap();
      llvm::SmallVector<AffineExpr> lbExprs;
      llvm::SmallVector<AffineExpr> ubExprs;
      llvm::SmallVector<AffineExpr> stepExprs;
      for (auto expr : ubMap.getResults()) {
        if (expr.isa<AffineDimExpr>()) {
          auto dimExpr = expr.cast<AffineDimExpr>();
          auto dim = dimExpr.getPosition();
          if (minifiedDimensions.count(dim) == 0)
            ubExprs.push_back(expr);
          else
            ubExprs.push_back(getAffineConstantExpr(minifiedDimensions[dim],
                                                    op->getContext()));
        } else if (expr.isa<AffineConstantExpr>()) {
          auto dim = expr.cast<AffineConstantExpr>().getValue();
          if (minifiedDimensions.count(dim) == 0)
            ubExprs.push_back(expr);
          else
            ubExprs.push_back(getAffineConstantExpr(minifiedDimensions[dim],
                                                    op->getContext()));
        } else if (expr.isa<AffineBinaryOpExpr>()) {
        } else {
          llvm::outs() << "expr type: " << (unsigned int)expr.getKind() << "\n";
          assert(false && "Unexpected expression type");
        }
      }
      auto ubMapMinified = AffineMap::get(
          ubMap.getNumDims(), ubMap.getNumSymbols(), ubExprs, op->getContext());

      forOp.setUpperBoundMap(ubMapMinified);

      if (debug) {
        llvm::outs() << "Loop bounds: " << ubMap << "\n";
        llvm::outs() << "Minified loop bounds: " << ubMapMinified << "\n";
      }
    }
  });
}

void MemrefMinifyPass::runOnOperation() {
  auto operation = getOperation();

  auto minifiedDimensions = getMinifedDimensionMap(operation);

  minifyMemrefs(operation, minifiedDimensions);
  minifyLoopBounds(operation, minifiedDimensions);
}

std::unique_ptr<OperationPass<ModuleOp>> createMemrefMinifyPass() {
  return std::make_unique<MemrefMinifyPass>();
}
