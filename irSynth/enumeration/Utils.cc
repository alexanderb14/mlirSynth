#include "Utils.h"

using namespace mlir;

Operation *createDummyOperation(MLIRContext &ctx, OperationName &opName) {
  OpBuilder builder1(&ctx);
  Operation *op1 = builder1.create(UnknownLoc::get(&ctx),
                                   opName.getIdentifier(), {}, {}, {});
  return op1;
}

int getRequiredNumOperands(mlir::Operation *op) {
  if (op->hasTrait<OpTrait::ZeroOperands>())
    return 0;
  if (op->hasTrait<OpTrait::OneOperand>())
    return 1;
  if (op->hasTrait<OpTrait::NOperands<2>::Impl>())
    return 2;
  if (op->hasTrait<OpTrait::NOperands<3>::Impl>())
    return 3;
  if (op->hasTrait<OpTrait::NOperands<4>::Impl>())
    return 4;
  if (op->hasTrait<OpTrait::NOperands<5>::Impl>())
    return 5;

  if (op->hasTrait<OpTrait::AtLeastNOperands<1>::Impl>())
    return 1;
  if (op->hasTrait<OpTrait::AtLeastNOperands<2>::Impl>())
    return 2;

  // FIXME: Currently variadic operands are not supported.
  if (op->hasTrait<OpTrait::VariadicOperands>())
    return 2;

  llvm::outs() << "Unsupported number of operands in op: "
               << op->getName().getStringRef().str().c_str();
  assert(false);
}

std::vector<StringAttr>
getFilteredAttributeNames(RegisteredOperationName opName) {
  std::vector<StringAttr> filteredAttrNames;
  for (auto attrName : opName.getAttributeNames()) {
    if (attrName == "precision_config" || attrName == "broadcast_dimensions" ||
        attrName == "dot_dimension_numbers")
      continue;
    filteredAttrNames.push_back(attrName);
  }
  return filteredAttrNames;
}

int getRequiredNumAttributes(mlir::Operation *op) {
  return getFilteredAttributeNames(op->getRegisteredInfo().value()).size();
}

int getRequiredNumRegions(mlir::Operation *op) {
  if (op->hasTrait<OpTrait::ZeroRegions>())
    return 0;

  if (op->hasTrait<OpTrait::OneRegion>())
    return 1;
  if (op->hasTrait<OpTrait::NRegions<2>::Impl>())
    return 2;
  if (op->hasTrait<OpTrait::NRegions<3>::Impl>())
    return 3;

  // FIXME: Currently variadic operands are not supported.
  if (op->hasTrait<OpTrait::VariadicRegions>())
    return 1;

  llvm::outs() << "Unsupported number of regions in op: "
               << op->getName().getStringRef().str().c_str();
  assert(false);
}

int getRequiredNumResults(mlir::Operation *op) {
  if (op->hasTrait<OpTrait::ZeroResults>())
    return 0;

  if (op->hasTrait<OpTrait::OneResult>())
    return 1;
  if (op->hasTrait<OpTrait::NResults<2>::Impl>())
    return 2;
  if (op->hasTrait<OpTrait::NResults<3>::Impl>())
    return 3;

  // FIXME: Currently variadic operands are not supported.
  if (op->hasTrait<OpTrait::VariadicResults>())
    return 1;

  llvm::outs() << "Unsupported number of results in op: "
               << op->getName().getStringRef().str().c_str();
  assert(false);
}

std::tuple<int, int>
getRequiredNumOperandsAndNumResults(MLIRContext &ctx, OperationName &opName) {
  Operation *dummyOp = createDummyOperation(ctx, opName);
  int numOperands = getRequiredNumOperands(dummyOp);
  int numResults = getRequiredNumResults(dummyOp);
  free(dummyOp);

  return {numOperands, numResults};
}
