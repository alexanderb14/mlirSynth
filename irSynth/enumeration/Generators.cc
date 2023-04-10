#include "Generators.h"

#include "enumeration/Grammar.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <llvm-14/llvm/ADT/ArrayRef.h>
#include <random>
#include <set>

using namespace mlir;

// Utility functions
// -----------------------------------------------------------------------------
int randomInteger(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  return dis(gen);
}

void printAttributes(
    std::vector<std::pair<Attribute, grammar::OpAndResType>> &attributes) {
  llvm::outs() << "Attributes:"
               << "\n--------\n";
  for (auto attr : attributes) {
    attr.first.dump();
    llvm::outs() << opAndResTypeToString(attr.second) << "\n";
    llvm::outs() << "---------\n";
  }
}

// Initial candidate generators
// -----------------------------------------------------------------------------
std::vector<CandidatePtr> InitialCandidateGenerator::gen() {
  std::vector<CandidatePtr> candidates;

  OpBuilder builder(&ctx);

  // Constant candidates.
  for (auto &attributePair : genAttributes(ctx, functionArgs, targetShape)) {
    auto &attr = attributePair.first;
    auto &type = attributePair.second;

    CandidatePtr candidate(new Candidate({}, type));
    candidate->addOperation(
        ctx, builder.create<stablehlo::ConstantOp>(UnknownLoc::get(&ctx), attr),
        false);
    candidates.push_back(candidate);
  }

  // Argument candidates.
  std::vector<mlir::Type> inputs;
  for (auto arg : functionArgs) {
    if (arg.getType().isa<ShapedType>()) {
      auto shapedType = arg.getType().cast<ShapedType>();
      inputs.push_back(RankedTensorType::get(shapedType.getShape(),
                                             shapedType.getElementType()));
    } else if (arg.getType().isa<FloatType>()) {
      inputs.push_back(RankedTensorType::get({}, arg.getType()));
    } else {
      llvm::outs() << "Type: " << arg.getType() << "\n";
      assert(false && "Unsupported type");
    }
  }

  unsigned argId = 0;
  for (auto &input : inputs) {
    CandidatePtr candidate(
        new Candidate({}, grammar::OpAndResType::HLO_Tensor));
    candidate->addArgument(ctx, input, argId++);
    candidates.push_back(candidate);
  }

  return candidates;
}

// Attribute generators
// -----------------------------------------------------------------------------
std::vector<::llvm::SmallVector<int64_t>>
genShapes(Region::BlockArgListType &functionArgs) {
  std::vector<::llvm::SmallVector<int64_t>> ret;

  for (auto arg : functionArgs) {
    if (!arg.getType().isa<ShapedType>())
      continue;

    auto shape = arg.getType().cast<ShapedType>().getShape();

    // Same shape as the argument: E.g. [3, 5, 7] -> [3, 5, 7]
    ret.emplace_back(shape.begin(), shape.end());

    // Dimension at previous last inserted: E.g. [3, 5, 7] -> [3, 5, 1, 7]
    ::llvm::SmallVector<int64_t> shapeWithOne(shape.begin(), shape.end());
    shapeWithOne.insert(shapeWithOne.end() - 1, 1);
    ret.emplace_back(shapeWithOne);

    // Leading dimension inserted: E.g. [5] -> [1, 5] or [3, 5] -> [1, 3, 5]
    ::llvm::SmallVector<int64_t> shapeWithOneLeading(shape.begin(),
                                                     shape.end());
    shapeWithOneLeading.insert(shapeWithOneLeading.begin(), 1);
    ret.emplace_back(shapeWithOneLeading);

    // Trailing dimension inserted: E.g. [5] -> [5, 1] or [3, 5] -> [3, 5, 1]
    ::llvm::SmallVector<int64_t> shapeWithOneTrailing(shape.begin(),
                                                      shape.end());
    shapeWithOneTrailing.insert(shapeWithOneTrailing.end(), 1);
    ret.emplace_back(shapeWithOneTrailing);

    // Transpose: E.g. [3, 5] -> [5, 3]
    ::llvm::SmallVector<int64_t> shapeTransposed(shape.begin(), shape.end());
    std::reverse(shapeTransposed.begin(), shapeTransposed.end());
    ret.emplace_back(shapeTransposed);

    // Reverse: E.g. when rank 1, [1, 0] or rank 2, [2, 1, 0]
    llvm::SmallVector<int64_t> sequenceReverse;
    for (int i = shape.size() - 1; i >= 0; i--) {
      sequenceReverse.push_back(i);
    }
    ret.emplace_back(sequenceReverse);
  }

  return ret;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genShapeAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                   llvm::ArrayRef<int64_t> &targetShape) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;

  auto shapes = genShapes(functionArgs);
  for (auto &shape : shapes) {
    attributes.emplace_back(builder.getI64TensorAttr(shape),
                            grammar::OpAndResType::HLO_DimensionTensor);
  }

  return attributes;
}

std::vector<int64_t> genUnaries(Region::BlockArgListType &functionArgs,
                                llvm::ArrayRef<int64_t> &targetShape) {
  std::set<int64_t> scalars;

  // Add 0 and 1.
  scalars.insert(0);
  scalars.insert(1);

  for (auto arg : functionArgs) {
    if (!arg.getType().isa<ShapedType>())
      continue;
    auto shape = arg.getType().cast<ShapedType>().getShape();

    for (unsigned i = 0; i < shape.size(); i++) {
      scalars.insert(i);
    }
  }

  return std::vector<int64_t>(scalars.begin(), scalars.end());
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genUnaryAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                   llvm::ArrayRef<int64_t> &targetShape) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;

  auto unaries = genUnaries(functionArgs, targetShape);
  for (auto &unary : unaries) {
    Attribute unaryAttr = builder.getF64FloatAttr((double)unary);
    Type type = RankedTensorType::get({}, builder.getF64Type());
    auto attrDense = DenseElementsAttr::get(type, {unaryAttr});

    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_Tensor);
  }

  return attributes;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genMaskAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                  llvm::ArrayRef<int64_t> &targetShape) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;

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
    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_PredTensor);

    // Create a matrix with 0 values.
    attrVect = std::vector<Attribute>();
    for (int i = 0; i < targetShape[0] * targetShape[1]; i++) {
      attrVect.push_back(builder.getF64FloatAttr(0.0));
    }
    type = RankedTensorType::get({targetShape[0], targetShape[1]},
                                 builder.getF64Type());
    attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attrVect);
    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_Tensor);
  }

  return attributes;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genAttributes(MLIRContext &ctx, Region::BlockArgListType &functionArgs,
              llvm::ArrayRef<int64_t> &targetShape) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;
  OpBuilder builder(&ctx);

  auto shapeAttributes = genShapeAttributes(builder, functionArgs, targetShape);
  attributes.insert(attributes.end(), shapeAttributes.begin(),
                    shapeAttributes.end());

  auto maskAttributes = genMaskAttributes(builder, functionArgs, targetShape);
  attributes.insert(attributes.end(), maskAttributes.begin(),
                    maskAttributes.end());

  auto unaryAttributes = genUnaryAttributes(builder, functionArgs, targetShape);
  attributes.insert(attributes.end(), unaryAttributes.begin(),
                    unaryAttributes.end());

  // printAttributes(attributes);

  return attributes;
}

std::vector<mlir::Attribute> AttributeGenerator::genDenseIntElementsAttr() {
  std::vector<mlir::Attribute> attributes;

  OpBuilder builder(&ctx);
  auto shapeAttributes = genAttributes(ctx, functionArgs, targetShape);
  for (auto attr : shapeAttributes) {
    attributes.push_back(attr.first);
  }

  return attributes;
}

std::vector<::llvm::SmallVector<int64_t>>
AttributeGenerator::genLlvmSmallVectorint64t() {
  std::vector<::llvm::SmallVector<int64_t>> attributes;

  attributes.emplace_back();

  auto unaries = genUnaries(functionArgs, targetShape);
  for (auto &unary : unaries) {
    ::llvm::SmallVector<int64_t> unaryAttr = {unary};
    attributes.push_back(unaryAttr);
  }

  return attributes;
}

// Region generators
// -----------------------------------------------------------------------------
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
  auto addOp = builder.create<stablehlo::AddOp>(
      unknownLoc, block->getArgument(0), block->getArgument(1));
  block->push_back(addOp);

  // Create a stablehlo return operation with the result of the add operation.
  auto returnOp = builder.create<stablehlo::ReturnOp>(
      unknownLoc, block->back().getResults());
  block->push_back(returnOp);

  std::vector<std::shared_ptr<Region>> regions;
  regions.push_back(region);

  return regions;
}
