#include "ChangeSizesPass.h"

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

using SizeMap = llvm::DenseMap<int64_t, int64_t>;

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

SizeMap getMinifedSizeMap(func::FuncOp &func) {
  bool debug = false;

  // Collect all memref types.
  llvm::SetVector<MemRefType> memrefTypes;
  func->walk([&](Operation *op) {
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
  SizeMap minifiedDimensions;
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

void changeTensorSizes(func::FuncOp &func,
                   SizeMap &minifiedSizes) {
}


void changeMemrefSizes(func::FuncOp &func,
                   SizeMap &newSizes) {
  // In function signatures.
  auto type = func.getFunctionType();
  // - Minify memref types in function arguments.
  llvm::SmallVector<Type> newArgTypes;
  for (auto argType : type.getInputs()) {
    if (argType.isa<MemRefType>()) {
      auto memrefType = argType.cast<MemRefType>();
      llvm::SmallVector<int64_t> newShape;
      for (auto dim : memrefType.getShape()) {
        long newDim;
        if (newSizes.count(dim) == 0)
          newDim = dim;
        else
          newDim = newSizes[dim];
        newShape.push_back(newDim);
      }
      auto newType =
          MemRefType::get(newShape, memrefType.getElementType());
      newArgTypes.push_back(newType);
    } else {
      newArgTypes.push_back(argType);
    }
  }

  // - Minify memref types in function results.
  llvm::SmallVector<Type> newResultTypes;
  for (auto resultType : type.getResults()) {
    if (resultType.isa<MemRefType>()) {
      auto memrefType = resultType.cast<MemRefType>();
      llvm::SmallVector<int64_t> newShape;
      for (auto dim : memrefType.getShape()) {
        long newDim;
        if (newSizes.count(dim) == 0)
          newDim = dim;
        else
          newDim = newSizes[dim];
        newShape.push_back(newDim);
      }
      auto newType =
          MemRefType::get(newShape, memrefType.getElementType());
      newResultTypes.push_back(newType);
    } else {
      newResultTypes.push_back(resultType);
    }
  }

  // - Set the new function type.
  auto newType = FunctionType::get(type.getContext(), newArgTypes,
                                        newResultTypes);
  func.setType(newType);

  // In operations.
  func->walk([&](Operation *op) {
    // - Minify memref types in operation operands.
    for (auto operand : op->getOperands()) {
      if (operand.getType().isa<MemRefType>()) {
        auto type = operand.getType().cast<MemRefType>();
        llvm::SmallVector<int64_t> newShape;
        for (auto dim : type.getShape()) {
          long newDim;
          if (newSizes.count(dim) == 0)
            newDim = dim;
          else
            newDim = newSizes[dim];
          newShape.push_back(newDim);
        }
        auto newType =
            MemRefType::get(newShape, type.getElementType());
        operand.setType(newType);
      }
    }
    // - Minify memref types in operation results.
    for (auto type : op->getResultTypes()) {
      if (type.isa<MemRefType>()) {
        auto resType = type.cast<MemRefType>();
        llvm::SmallVector<int64_t> newShape;
        for (auto dim : resType.getShape()) {
          long newDim;
          if (newSizes.count(dim) == 0)
            newDim = dim;
          else
            newDim = newSizes[dim];
          newShape.push_back(newDim);
        }
        auto newType =
            MemRefType::get(newShape, resType.getElementType());
        resType = newType;
      }
    }
  });
}

void changeLoopBounds(func::FuncOp &func,
                      SizeMap &newSizes) {
  bool debug = false;

  func->walk([&](Operation *op) {
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
          if (newSizes.count(dim) == 0)
            ubExprs.push_back(expr);
          else
            ubExprs.push_back(getAffineConstantExpr(newSizes[dim],
                                                    op->getContext()));
        } else if (expr.isa<AffineConstantExpr>()) {
          auto dim = expr.cast<AffineConstantExpr>().getValue();
          if (newSizes.count(dim) == 0)
            ubExprs.push_back(expr);
          else
            ubExprs.push_back(getAffineConstantExpr(newSizes[dim],
                                                    op->getContext()));
        } else if (expr.isa<AffineBinaryOpExpr>()) {
        } else {
          llvm::outs() << "expr type: " << (unsigned int)expr.getKind() << "\n";
          assert(false && "Unexpected expression type");
        }
      }
      auto ubMapNew = AffineMap::get(
          ubMap.getNumDims(), ubMap.getNumSymbols(), ubExprs, op->getContext());
      if (ubMapNew.getNumResults())
        forOp.setUpperBoundMap(ubMapNew);

      if (debug) {
        llvm::outs() << "Loop bounds: " << ubMap << "\n";
        llvm::outs() << "Minified loop bounds: " << ubMapNew << "\n";
      }
    }
  });
}

void annotateChangedSizes(
    func::FuncOp &func, SizeMap &newSizes) {
  std::string newSizesStr;
  bool first = true;
  for (auto dim : newSizes) {
    if (!first)
      newSizesStr += ",";
    first = false;
    newSizesStr +=
        std::to_string(dim.first) + ":" + std::to_string(dim.second);
  }
  func->setAttr("changed_sizes",
                StringAttr::get(func->getContext(), newSizesStr));
}

SizeMap getChangedSizes(func::FuncOp &func) {
  SizeMap changedSizes;

  auto changedSizesAttr = func->getAttr("changed_sizes");
  assert(changedSizesAttr && "No changed_sizes attribute found");
  auto changedSizesStr = changedSizesAttr.cast<StringAttr>().getValue().str();

  std::stringstream ss(changedSizesStr);
  std::string token;
  while (std::getline(ss, token, ',')) {
    std::stringstream ss2(token);
    std::string dimStr;
    std::string newDimStr;
    std::getline(ss2, dimStr, ':');
    std::getline(ss2, newDimStr, ':');
    long dim = std::stol(dimStr);
    long newDim = std::stol(newDimStr);
    changedSizes[newDim] = dim;
  }

  return changedSizes;
}

void ChangeSizesPass::runOnOperation() {
  auto operation = getOperation();

  if (mode == "minify") {
    operation->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        auto func = cast<func::FuncOp>(op);

        auto minifiedSizes = getMinifedSizeMap(func);
        changeMemrefSizes(func, minifiedSizes);
        changeLoopBounds(func, minifiedSizes);

        annotateChangedSizes(func, minifiedSizes);
      }
    });
  } else if (mode == "restore") {
    operation->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        auto func = cast<func::FuncOp>(op);

        auto changedSizes = getChangedSizes(func);
        changeMemrefSizes(func, changedSizes);
        changeTensorSizes(func, changedSizes);
        changeLoopBounds(func, changedSizes);
      }
    });

  } else {
    llvm::outs() << "Unknown mode: " << mode << "\n";
    assert(false && "Unknown mode");
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createChangeSizesPass() {
  return std::make_unique<ChangeSizesPass>();
}
