#include "Generators.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

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

std::vector<Attribute>
genShapeAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs) {
  std::vector<Attribute> attributes;

  for (auto arg : functionArgs) {
    if (!arg.getType().isa<ShapedType>())
      continue;

    auto shape = arg.getType().cast<ShapedType>().getShape();

    // Same shape as the argument: E.g. [3, 5, 7] -> [3, 5, 7]
    auto attrVect = std::vector<Attribute>();
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
    }
    attributes.push_back(getDenseElementsAttr(attrVect));

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
    attributes.push_back(getDenseElementsAttr(attrVect));

    // Leading dimension inserted: E.g. [5] -> [1, 5] or [3, 5] -> [1, 3, 5]
    attrVect = std::vector<Attribute>();
    attrVect.push_back(builder.getI64IntegerAttr(1));
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
    }
    attributes.push_back(getDenseElementsAttr(attrVect));

    // Trailing dimension inserted: E.g. [5] -> [5, 1] or [3, 5] -> [3, 5, 1]
    attrVect = std::vector<Attribute>();
    for (auto dim : shape) {
      attrVect.push_back(builder.getI64IntegerAttr(dim));
    }
    attrVect.push_back(builder.getI64IntegerAttr(1));
    attributes.push_back(getDenseElementsAttr(attrVect));
  }

  return attributes;
}

std::vector<Attribute>
genTensorAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                    int maxRank) {
  std::vector<Attribute> tensorValues;

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

void printAttributes(std::vector<Attribute>& attributes) {
  llvm::outs() << "Attributes:"
               << "\n--------\n";
  for (auto attr : attributes) {
    attr.dump();
  }
}

std::vector<Attribute>
genAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
              int maxRank) {
  std::vector<Attribute> attributes;

  auto shapeValues = genShapeAttributes(builder, functionArgs);
  attributes.insert(attributes.end(), shapeValues.begin(), shapeValues.end());

  auto tensorValues = genTensorAttributes(builder, functionArgs, maxRank);
  attributes.insert(attributes.end(), tensorValues.begin(), tensorValues.end());

  // printAttributes(attributes);

  return attributes;
}

std::vector<std::shared_ptr<Region>> genRegions(OpBuilder &builder) {
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
