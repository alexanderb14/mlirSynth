#include "Generators.h"

#include "enumeration/OpInfos.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"

#include <random>

using namespace mlir;

int randomInteger(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  return dis(gen);
}

DenseElementsAttr getDenseElementsAttr(std::vector<Attribute> attrVect) {
  Type type = RankedTensorType::get({static_cast<long>(attrVect.size())},
                                    attrVect[0].cast<TypedAttr>().getType());
  return DenseElementsAttr::get(type.cast<TensorType>(), attrVect);
}

std::vector<std::pair<Attribute, OpAndResType>>
genShapeAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs) {
  std::vector<std::pair<Attribute, OpAndResType>> attributes;

  for (auto arg : functionArgs) {
    if (!arg.getType().isa<ShapedType>())
      continue;

    auto shape = arg.getType().cast<ShapedType>().getShape();

    // Same shape as the argument: E.g. [3, 5, 7] -> [3, 5, 7]
    auto attrVect = std::vector<Attribute>();
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
    }
    attributes.emplace_back(getDenseElementsAttr(attrVect),
                            OpAndResType::HLO_DimensionTensor);

    // Dimension at previous last inserted: E.g. [3, 5, 7] -> [3, 5, 1, 7]
    attrVect = std::vector<Attribute>();
    unsigned dimIdx = 0;
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
      if (dimIdx == shape.size() - 2) {
        attrVect.push_back(builder.getI64IntegerAttr(1));
      }
      dimIdx++;
    }
    attributes.emplace_back(getDenseElementsAttr(attrVect),
                            OpAndResType::HLO_DimensionTensor);

    // Leading dimension inserted: E.g. [5] -> [1, 5] or [3, 5] -> [1, 3, 5]
    attrVect = std::vector<Attribute>();
    attrVect.push_back(builder.getI64IntegerAttr(1));
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
    }
    attributes.emplace_back(getDenseElementsAttr(attrVect),
                            OpAndResType::HLO_DimensionTensor);

    // Trailing dimension inserted: E.g. [5] -> [5, 1] or [3, 5] -> [3, 5, 1]
    attrVect = std::vector<Attribute>();
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
    }
    attrVect.push_back(builder.getI64IntegerAttr(1));
    attributes.emplace_back(getDenseElementsAttr(attrVect),
                            OpAndResType::HLO_DimensionTensor);

    // Transpose: E.g. [3, 5] -> [5, 3]
    attrVect = std::vector<Attribute>();
    for (unsigned i = 0; i < shape.size(); i++) {
      attrVect.push_back(builder.getI64IntegerAttr(i));
    }
    std::reverse(attrVect.begin(), attrVect.end());
    attributes.emplace_back(getDenseElementsAttr(attrVect),
                            OpAndResType::HLO_DimensionTensor);
  }

  return attributes;
}

std::vector<std::pair<Attribute, OpAndResType>>
genTensorAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                    llvm::ArrayRef<int64_t> &targetShape, int maxRank) {
  std::vector<std::pair<Attribute, OpAndResType>> tensorValues;

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
      tensorValues.emplace_back(attrDense,
                                OpAndResType::HLO_Tensor);
    }
  }

  if (targetShape.size() == 2) {
    // Create lower triangular mask containing of i1 values
    std::vector<Attribute> attrVect;
    for (int i = 0; i < targetShape[0]; i++) {
      for (int j = 0; j < targetShape[1]; j++) {
        if (i >= j) {
          attrVect.push_back(builder.getBoolAttr(true));
        } else {
          attrVect.push_back(builder.getBoolAttr(false));
        }
      }
    }
    Type type = RankedTensorType::get({targetShape[0], targetShape[1]},
                                      builder.getI1Type());
    auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attrVect);
    tensorValues.emplace_back(attrDense,
                              OpAndResType::HLO_PredTensor);

    // Create a matrix with 0 values.
    attrVect = std::vector<Attribute>();
    for (int i = 0; i < targetShape[0] * targetShape[1]; i++) {
      attrVect.push_back(builder.getF64FloatAttr(0.0));
    }
    type = RankedTensorType::get({targetShape[0], targetShape[1]},
                                 builder.getF64Type());
    attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attrVect);
    tensorValues.emplace_back(attrDense,
                              OpAndResType::HLO_Tensor);
  }

  return tensorValues;
}

void printAttributes(std::vector<std::pair<Attribute, OpAndResType>>& attributes) {
  llvm::outs() << "Attributes:"
               << "\n--------\n";
  for (auto attr : attributes) {
    attr.first.dump();
    llvm::outs() << opAndResTypeToString(attr.second) << "\n";
    llvm::outs() << "---------\n";
  }
}

std::vector<std::pair<Attribute, OpAndResType>>
genAttributes(MLIRContext &ctx, Region::BlockArgListType &functionArgs,
              llvm::ArrayRef<int64_t> &targetShape, int maxRank) {
  std::vector<std::pair<Attribute, OpAndResType>> attributes;
  OpBuilder builder(&ctx);

  auto shapeValues = genShapeAttributes(builder, functionArgs);
  attributes.insert(attributes.end(), shapeValues.begin(), shapeValues.end());

  auto tensorValues = genTensorAttributes(builder, functionArgs, targetShape, maxRank);
  attributes.insert(attributes.end(), tensorValues.begin(), tensorValues.end());

  // printAttributes(attributes);

  return attributes;
}

std::vector<std::shared_ptr<Region>> genRegions(MLIRContext &ctx) {
  OpBuilder builder(&ctx);
  auto unknownLoc = UnknownLoc::get(&ctx);

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
