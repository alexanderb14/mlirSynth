#include "Utils.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"

#include <random>

using namespace mlir;

int randomInteger(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  return dis(gen);
}

std::vector<func::FuncOp> getFunctions(mlir::Operation *op) {
  std::vector<func::FuncOp> functions;
  op->walk([&](func::FuncOp func) { functions.push_back(func); });
  return functions;
}

std::vector<Attribute>
getTensorAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                    int maxRank) {
  std::vector<Attribute> tensorValues;

  auto attr = std::vector<Attribute>();
  attr.push_back(builder.getI64IntegerAttr(5));
  attr.push_back(builder.getI64IntegerAttr(3));
  attr.push_back(builder.getI64IntegerAttr(1));
  attr.push_back(builder.getI64IntegerAttr(7));
  Type type = RankedTensorType::get({static_cast<long>(attr.size())},
                                    attr[0].cast<TypedAttr>().getType());
  auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
  tensorValues.push_back(attrDense);


  if (maxRank >= 0) {
    std::vector<Attribute> attrs = {
        //        builder.getBoolAttr(true),    builder.getBoolAttr(false),
        builder.getF64FloatAttr(0.0),
        builder.getF64FloatAttr(1.0),
        //        builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(1),
        IntegerAttr::get(
            IntegerType::get(builder.getContext(), 64, IntegerType::Signless),
            APInt(64, 0, false)),
    };
    for (auto attr : attrs) {
      Type type = RankedTensorType::get({}, attr.cast<TypedAttr>().getType());
      auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
      tensorValues.push_back(attrDense);
    }
  }

  if (maxRank >= 1) {
    std::vector<std::vector<Attribute>> attrs = {
        std::vector<Attribute>{builder.getF64FloatAttr(0)},
        std::vector<Attribute>{builder.getF64FloatAttr(1)}};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        attrs.push_back(std::vector<Attribute>{builder.getF64FloatAttr(i),
                                               builder.getF64FloatAttr(j)});
      }
    }
    for (auto attr : attrs) {
      Type type = RankedTensorType::get({static_cast<long>(attr.size())},
                                        attr[0].cast<TypedAttr>().getType());
      auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
      tensorValues.push_back(attrDense);
    }
  }

  if (maxRank >= 2) {
    int n = 2;
    std::vector<Attribute> attr;
    attr.reserve(n * n);
    for (int i = 0; i < n * n; i++) {
      attr.push_back(builder.getF64FloatAttr(randomInteger(0, 10)));
    }

    std::vector<std::vector<Attribute>> attrs;
    attrs.push_back(attr);

    for (auto attr : attrs) {
      Type type =
          RankedTensorType::get({n, n}, attr[0].cast<TypedAttr>().getType());
      auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
      tensorValues.push_back(attrDense);
    }
  }

  return tensorValues;
}

std::vector<std::shared_ptr<Region>> getRegions(OpBuilder &builder) {
  auto unknownLoc = builder.getUnknownLoc();

  // Create region with a single block.
  std::shared_ptr<Region> region = std::make_shared<Region>();
  Block *block = new Block();
  region->push_back(block);

  // Add two arguments to the block.
  auto tensorType = RankedTensorType::get({}, builder.getF64Type());
  block->addArgument(tensorType, unknownLoc);
  block->addArgument(tensorType, unknownLoc);

  // Create a add operation with the two arguments.
  auto addOp = builder.create<mhlo::AddOp>(unknownLoc, block->getArgument(0),
                                           block->getArgument(1));
  block->push_back(addOp);

  // Create a mhlo return operation with the result of the add operation.
  auto returnOp =
      builder.create<mhlo::ReturnOp>(unknownLoc, block->back().getResults());
  block->push_back(returnOp);

  std::vector<std::shared_ptr<Region>> regions;
  regions.push_back(region);

  return regions;
}

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
    if (attrName == "precision_config" || attrName == "broadcast_dimensions")
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
