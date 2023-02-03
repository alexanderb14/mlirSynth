#include "Enumerator.h"

#include "Common.h"
#include "enumeration/ArgTuples.h"
#include "enumeration/Candidate.h"
#include "enumeration/Generators.h"
#include "enumeration/Utils.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <math.h>
#include <optional>
#include <variant>

using namespace llvm;
using namespace mlir;

enum ProcessingStatus {
  reject_hasTooManyOps,
  reject_isNotResultTypeInferrable,
  reject_hasUnsupportedShapeRank,
  reject_isNotAllDefsAreUsed,
  reject_isNotVerifiable,
  reject_hasNoArguments,
  reject_hasUnknownRankAndShape,
  reject_isNotCompilableToLLVM,
  reject_hasEmptyReturnShape,
  reject_isNotExecutable,
  reject_hashNotUnique,
  accept_as_candidate,
  accept_as_solution,
};

std::string processingStatusToStr(ProcessingStatus &status) {
  if (status == reject_hasTooManyOps)
    return "reject_hasTooManyOps";
  if (status == reject_isNotResultTypeInferrable)
    return "reject_isNotResultTypeInferrable";
  if (status == reject_hasUnsupportedShapeRank)
    return "reject_hasUnsupportedShapeRank";
  if (status == reject_isNotAllDefsAreUsed)
    return "reject_isNotAllDefsAreUsed";
  if (status == reject_isNotVerifiable)
    return "reject_isNotVerifiable";
  if (status == reject_hasNoArguments)
    return "reject_hasNoArguments";
  if (status == reject_hasUnknownRankAndShape)
    return "reject_hasUnknownRankAndShape";
  if (status == reject_isNotCompilableToLLVM)
    return "reject_isNotCompilableToLLVM";
  if (status == reject_hasEmptyReturnShape)
    return "reject_hasEmptyReturnShape";
  if (status == reject_isNotExecutable)
    return "reject_isNotExecutable";
  if (status == reject_hashNotUnique)
    return "reject_hashNotUnique";
  if (status == accept_as_candidate)
    return "accept_as_candidate";
  if (status == accept_as_solution)
    return "accept_as_solution";
  assert(false && "Processing Status not known");
}

unsigned maxShapeRank = 4;

void printCandidate(ProcessingStatus status,
                    CandidateStorePtr &localCandidateStore,
                    CandidateStorePtr &candidateStore, CandidatePtr &candidate,
                    EnumerationOptions &options,
                    OwningOpRef<ModuleOp> &module) {
  // If there is nothing to print, return early.
  if (!(options.printStatusNames || options.printStatusTiles ||
        options.printValidCandidates || options.printInvalidCandidates)) {
    return;
  }

  // Build and print the status string.
  int candidateId = localCandidateStore->getCandidateId(candidate);

  std::string statusStr;
  bool printStatus = options.printStatusNames || options.printStatusTiles ||
                     options.printValidCandidates ||
                     options.printInvalidCandidates;
  if (printStatus) {
    if (options.printStatusTiles) {
      statusStr = " ";
    } else {
      statusStr = "Candidate " + std::to_string(candidateId) + ": ";

      statusStr += "status:" + processingStatusToStr(status);

      statusStr += ", preds:";
      bool first = true;
      for (auto &pred : candidate->getPredecessors()) {
        if (!first)
          statusStr += ",";
        first = false;
        statusStr += std::to_string(candidateStore->getCandidateId(pred));
      }
    }

    if (status == accept_as_candidate) {
      statusStr = "\033[1;42m" + statusStr + "\033[0m";
    } else {
      statusStr = "\033[1;41m" + statusStr + "\033[0m";
    }
  }

  // Print the module.
  if ((status == accept_as_candidate && options.printValidCandidates) ||
      (!(status == accept_as_candidate) && options.printInvalidCandidates) ||
      options.printStatusNames) {
    llvm::outs() << statusStr << "\n";
    if (status > reject_hasUnsupportedShapeRank)
      module->print(llvm::outs());
  }
}

void prepareInputFunction(func::FuncOp &inputFunction) {
  inputFunction->setAttr("llvm.emit_c_interface",
                         UnitAttr::get(inputFunction->getContext()));
  inputFunction.setName("foo");
}

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, func::FuncOp *function) {
  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();

  // Add the function to the module block.
  moduleBlock.push_back(function->getOperation()->clone());

  return module;
}

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, Region *region) {
  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Create function.
  auto func = builder.create<func::FuncOp>(
      unknownLoc, "foo", mlir::FunctionType::get(&ctx, {}, {}));

  // Add the given region to the function.
  BlockAndValueMapping mapper;
  region->cloneInto(&func.getFunctionBody(), mapper);

  auto *bodyBlock = &func.getFunctionBody().getBlocks().front();
  builder.setInsertionPoint(bodyBlock, bodyBlock->end());

  // Add return to the function.
  if (bodyBlock->empty()) {
    // Is argument.
    builder.create<func::ReturnOp>(unknownLoc, bodyBlock->getArguments());
    func.setFunctionType(mlir::FunctionType::get(
        &ctx, bodyBlock->getArgumentTypes(), bodyBlock->getArgumentTypes()));
  } else {
    // Is operations.
    auto &lastOp = bodyBlock->back();
    builder.create<func::ReturnOp>(unknownLoc, lastOp.getResults());
    func.setFunctionType(mlir::FunctionType::get(
        &ctx, bodyBlock->getArgumentTypes(), lastOp.getResultTypes()));
  }

  // // Print operands per operation.
  // llvm::outs() << "-------------------------------\n";
  // func.walk([&](Operation *op) {
  //   llvm::outs() << op->getName() << " " << op->getNumOperands() << "\n";
  //   for (auto &operand : op->getOpOperands()) {
  //     llvm::outs() << "  " << operand.get() << "\n";
  //   }
  // });
  // llvm::outs() << "-------------------------------\n";

  return module;
}

LogicalResult inferResultTypes(MLIRContext &ctx, Operation *op,
                               RegisteredOperationName &opName) {
  // Not all operations have the infer return types function.
  // Therefore, we implement some manually.
  auto opNameStr = op->getName().getStringRef().str();
  if (opNameStr == "mhlo.dynamic_reshape") {
    // Construct a tensor type with the shape values of the second operand.
    SmallVector<int64_t, 4> shape;
    auto *argOp = op->getOperand(1).getDefiningOp();
    if (argOp) {
      auto constantOp = dyn_cast<mhlo::ConstantOp>(argOp);
      if (constantOp) {
        auto denseAttr = constantOp.getValue().dyn_cast<DenseElementsAttr>();
        if (denseAttr) {
          // Check if the dense attribute is a vector.
          if (denseAttr.getType().getRank() == 1) {
            // Get the shape values and add them to the shape vector.
            for (auto value : denseAttr.getValues<IntegerAttr>()) {
              shape.push_back(value.getValue().getSExtValue());
            }
          }
        }
      }
    }

    auto arg0Type = op->getOperand(0).getType();
    auto newTensorType = RankedTensorType::get(
        shape, arg0Type.cast<TensorType>().getElementType());
    op->getResult(0).setType(newTensorType);

    return success();
  }
  if (opNameStr == "mhlo.dot_general") {
    // Get the return shapes of the lhs and rhs operands.
    auto lhsShape =
        op->getOperand(0).getType().cast<RankedTensorType>().getShape();
    auto rhsShape =
        op->getOperand(1).getType().cast<RankedTensorType>().getShape();

    // Get the contraction dimensions.
    auto dotDimensionNumbersAttr =
        op->getAttrOfType<mhlo::DotDimensionNumbersAttr>(
            "dot_dimension_numbers");
    auto lhsContractingDimensions =
        dotDimensionNumbersAttr.getLhsContractingDimensions();
    auto rhsContractingDimensions =
        dotDimensionNumbersAttr.getRhsContractingDimensions();
    assert(lhsContractingDimensions.size() == 1);
    assert(rhsContractingDimensions.size() == 1);
    auto lhsContractingDimension = lhsContractingDimensions[0];
    auto rhsContractingDimension = rhsContractingDimensions[0];

    // The return shape is the concatenation of the lhs and rhs shapes along lhs
    // and rhs contraction dimensions.
    SmallVector<int64_t, 4> shape;
    for (unsigned i = 0; i < lhsContractingDimension; i++) {
      shape.push_back(lhsShape[i]);
    }
    for (unsigned i = rhsContractingDimension + 1; i < rhsShape.size(); i++) {
      shape.push_back(rhsShape[i]);
    }

    auto arg0Type = op->getOperand(0).getType();
    auto newTensorType = RankedTensorType::get(
        shape, arg0Type.cast<TensorType>().getElementType());
    op->getResult(0).setType(newTensorType);

    return success();
  }

  // Check if the operation is well-formed. If not, bail out
  // early because otherwise type inference will assert.
  std::vector<std::string> verifyTraitsOnlyOps = {"mhlo.dot"};
  std::string opname = op->getName().getStringRef().str();
  bool verifyTraitsOnly =
      std::find(verifyTraitsOnlyOps.begin(), verifyTraitsOnlyOps.end(),
                opname) != verifyTraitsOnlyOps.end();

  if (verifyTraitsOnly) {
    if (failed(opName.verifyTraits(op))) {
      return failure();
    }
  } else {
    if (failed(opName.verifyInvariants(op))) {
      return failure();
    }
  }

  // Infer the type.
  SmallVector<mlir::Type> inferredTypes;
  if (auto inferResultTypes = dyn_cast<InferTypeOpInterface>(op)) {
    if (failed(inferResultTypes.inferReturnTypes(
            &ctx, op->getLoc(), op->getOperands(), op->getAttrDictionary(),
            op->getRegions(), inferredTypes))) {
      return failure();
    }
  }

  // Check if the inferred type is valid.
  if (inferredTypes.size() != op->getNumResults()) {
    llvm::outs() << "Inferred type size does not match the number of results."
                 << "inferredTypes.size() = " << inferredTypes.size()
                 << ", op->getNumResults() = " << op->getNumResults() << "\n";
    assert(false);
  }
  for (auto &type : inferredTypes) {
    if (!type) {
      return failure();
    }
  }

  // Set the inferred result types.
  for (unsigned i = 0; i < op->getNumResults(); i++) {
    op->getResult(i).setType(inferredTypes[i]);
  }

  return success();
}

void initializeCandidates(MLIRContext &ctx, CandidateStorePtr &candidateStore,
                          Region::BlockArgListType functionArgs) {
  OpBuilder builder(&ctx);

  // Constant candidates.
  for (auto &attr : genAttributes(builder, functionArgs, 0)) {
    CandidatePtr candidate(new Candidate({}));
    candidate->addOperation(
        ctx, builder.create<mhlo::ConstantOp>(UnknownLoc::get(&ctx), attr),
        false);
    candidateStore->addCandidate(candidate, 0);
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
    CandidatePtr candidate(new Candidate({}));
    candidate->addArgument(ctx, input, argId++);
    candidateStore->addCandidate(candidate, 0);
  }
}

bool verifyDefsAreUsed(Block *block) {
  mlir::DenseMap<mlir::Value, bool> values;

  block->walk([&](Block *block) {
    for (auto &arg : block->getArguments()) {
      values[arg] = false;
    }
    for (auto &op : block->getOperations()) {
      for (auto result : op.getResults()) {
        values[result] = false;
      }
      for (auto operand : op.getOperands()) {
        values[operand] = true;
      }
    }
  });

  for (auto &value : values) {
    if (!value.second) {
      return false;
    }
  }
  return true;
}

bool hasTargetShape(Operation *op, ArrayRef<int64_t> targetShape) {
  auto shape = op->getResult(0).getType().cast<ShapedType>().getShape();
  return shape == targetShape;
}

bool hasRankedAndKnownShape(Operation *op) {
  auto shapedType = op->getResult(0).getType().cast<ShapedType>();
  return shapedType.hasStaticShape();
}

ProcessingStatus
process(MLIRContext &ctx, EnumerationStats &stats,
        RegisteredOperationName &opName, IExecutorPtr &executor,
        std::vector<ReturnAndArgType> &args, CandidateStorePtr &candidateStore,
        CandidateStorePtr &localCandidateStore, double *refOut,
        EnumerationOptions &options, ArgTuple operandArgTuple,
        CandidatePtr &newCandidate, OwningOpRef<ModuleOp> &module,
        ArrayRef<int64_t> &targetShape) {
  stats.numEnumerated++;

  // Create candidate.
  newCandidate.reset(new Candidate(operandArgTuple.operands));
  auto builder = OpBuilder(&ctx);

  // Set up operands.
  SmallVector<mlir::Value> operands =
      newCandidate->merge(ctx, operandArgTuple.operands);

  // Set up attributes.
  auto attrNames = getFilteredAttributeNames(opName);
  auto attrValues = operandArgTuple.attributes;
  assert(attrNames.size() == attrValues.size() &&
         "Attribute names and values must have the same size.");

  SmallVector<NamedAttribute> attributes = {};
  for (unsigned i = 0; i < attrNames.size(); i++) {
    StringAttr attrName = attrNames[i];
    mlir::Attribute value = attrValues[i];
    attributes.push_back(builder.getNamedAttr(attrName, value));
  }

  auto unfilteredAttrNames = opName.getAttributeNames();
  for (auto attrName : unfilteredAttrNames) {
     if (attrName.str() == "dot_dimension_numbers") {
      // Last element of lhsOpShape is the dimension to be contracted
      auto lhsOpShape = operands[0].getType().cast<ShapedType>().getShape();
      int64_t lhsContracting = lhsOpShape.size() - 1;
      // First element of rhsOpShape is the dimension to be contracted
      int64_t rhsContracting = 0;

      auto dotDimensionNumbers = mhlo::DotDimensionNumbersAttr::get(
          &ctx, {}, {}, {lhsContracting}, {rhsContracting});
      attributes.push_back(builder.getNamedAttr(attrName, dotDimensionNumbers));
    }
  }

  // Set up regions.
  SmallVector<std::unique_ptr<Region>> regions = {};
  for (auto &regionCandidate : operandArgTuple.regions) {
    std::unique_ptr<Region> region = std::make_unique<Region>();
    BlockAndValueMapping mapping;
    regionCandidate->cloneInto(region.get(), mapping);
    regions.push_back(std::move(region));
  }

  // Set up results types.
  // TODO: Parse number of results from the op definition.
  SmallVector<mlir::Type> resultTypes;
  if (operands.empty()) {
    resultTypes.push_back(builder.getNoneType());
  } else {
    resultTypes.push_back(operands[0].getType());
  }

  // Create operation.
  Operation *op =
      builder.create(UnknownLoc::get(&ctx), opName.getIdentifier(), operands,
                     resultTypes, attributes, {}, regions);
  newCandidate->addOperation(ctx, op);

  // Check length.
  if (newCandidate->getNumOps() > options.maxNumOps) {
    return reject_hasTooManyOps;
  }

  // Infer the operation result type.
  if (failed(inferResultTypes(ctx, op, opName))) {
    if (options.printInvalidCandidates)
      createModule(ctx, newCandidate->getRegion())->dump();
    return reject_isNotResultTypeInferrable;
  }

  // Check if the operation result shape rank is supported.
  for (auto resultType : op->getResultTypes()) {
    if (resultType.isa<RankedTensorType>()) {
      auto shape = resultType.cast<RankedTensorType>().getShape();
      if (shape.size() > maxShapeRank) {
        return reject_hasUnsupportedShapeRank;
      }
    }
  }

  // Verify candidate.
  module = createModule(ctx, newCandidate->getRegion());

  if (!verifyDefsAreUsed(&module->getRegion().getBlocks().front())) {
    return reject_isNotAllDefsAreUsed;
  }
  if (!succeeded(verify(*module))) {
    return reject_isNotVerifiable;
  }
  if (newCandidate->getNumArguments() == 0) {
    return reject_hasNoArguments;
  }
  if (!hasRankedAndKnownShape(op)) {
    return reject_hasUnknownRankAndShape;
  }

  stats.numValid++;
  auto returnShape = op->getResult(0).getType().cast<ShapedType>().getShape();

  auto func = module->lookupSymbol<func::FuncOp>("foo");
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(&ctx));

  ModuleOp moduleCopy = copyModuleToCtx(&ctx, module.get());
  if (failed(executor->lowerCHLOToLLVMDialect(moduleCopy))) {
    return reject_isNotCompilableToLLVM;
  }

  if (returnShape.empty())
    return reject_hasEmptyReturnShape;

  // Create args array.
  auto argsCand = selectArgs(args, newCandidate->getArgIds());
  auto returnShapeCand = getReturnShape(func);
  auto retCand = getOwningMemRefForShape(returnShapeCand);

  // Compile and run.
  if (failed(jitAndInvoke(moduleCopy, argsCand, retCand, true)))
    return reject_isNotExecutable;
  stats.numExecuted++;

  double *out = getReturnDataPtr(retCand);
  // printArray(out, returnShape);

  // Hash and add to store if hash doesn't exist yet.
  double hash = hashArray(out, returnShape);
  newCandidate->setHash(hash);
  //llvm::outs() << "Hash: " << hash << "\n";
  if (options.ignoreEquivalentCandidates &&
      !candidateStore->addCandidateHash(hash)) {
    stats.numIgnored++;
    return reject_hashNotUnique;
  }

  localCandidateStore->addCandidate(newCandidate, newCandidate->getNumOps());

  if (returnShape == targetShape) {
    if (areArraysEqual(refOut, out, returnShape)) {
      LLVM_DEBUG(llvm::dbgs() << "Found a match!\n");
      LLVM_DEBUG(module->print(llvm::dbgs()));
      // printArray(out, returnShape);

      candidateStore->merge(localCandidateStore);
      stats.numOps = newCandidate->getNumOps();

      if (options.printStats) {
        stats.dump();
      }

      return accept_as_solution;
    }
  }

  return accept_as_candidate;
}

ModuleAndArgIds
enumerateCandidates(MLIRContext &ctx, IExecutorPtr executor,
                    func::FuncOp inputFunction,
                    CandidateStorePtr &candidateStore,
                    std::vector<RegisteredOperationName> &avaliableOps,
                    EnumerationOptions &options) {
  auto inputFunctionName = inputFunction.getName().str();
  auto targetShape = getReturnShape(inputFunction);
  prepareInputFunction(inputFunction);

  // Compile and run reference.
  // - Create argument vector.
  auto args = createArgs(inputFunction.getArguments());
  randomlyInitializeArgs(args);
  if (options.printArgsAndResults)
    printArgs(args, llvm::outs());
  auto ret = getOwningMemRefForShape(targetShape);

  // - Run on argument vector gives the reference out.
  auto inputModuleRef = createModule(ctx, &inputFunction);
  auto inputModule = inputModuleRef.release();
  assert(succeeded(executor->lowerAffineToLLVMDialect(inputModule)));
  assert(succeeded(jitAndInvoke(inputModule, args, ret, false)));

  double *refOut = getReturnDataPtr(ret);
  if (options.printArgsAndResults)
    printArray(refOut, targetShape, llvm::outs());

  convertScalarToMemrefArgs(args);

  // Synthesize.
  // - Initialize candidate store with constant and argument candidates.
  initializeCandidates(ctx, candidateStore, inputFunction.getArguments());
  // - Print them.
  for (auto &candidate : candidateStore->getCandidates()) {
    auto module = createModule(ctx, candidate->getRegion());
    printCandidate(ProcessingStatus::accept_as_candidate, candidateStore,
                   candidateStore, candidate, options, module);
  }

  OwningOpRef<ModuleOp> acceptedModule = nullptr;
  CandidatePtr acceptedCandidate = nullptr;

  // - Enumerate candidates.
  EnumerationStats stats;
  for (int numOps = 0; numOps <= options.maxNumOps; numOps++) {
    CandidateStorePtr localCandidateStore = std::make_shared<CandidateStore>();

    for (auto opName : avaliableOps) {
      auto operandCandidates = candidateStore->getCandidates(numOps);
      auto operandArgTuples =
          getOperandArgTuples(ctx, opName, operandCandidates);

      auto status = failableParallelForEach(
          &ctx, operandArgTuples, [&](auto &operandArgTuple) {
            CandidatePtr newCandidate;
            OwningOpRef<ModuleOp> module;

            ProcessingStatus status =
                process(ctx, stats, opName, executor, args, candidateStore,
                        localCandidateStore, refOut, options,
                        operandArgTuple, newCandidate, module, targetShape);

            if (status == accept_as_solution) {
              acceptedModule = std::move(module);
              acceptedCandidate = newCandidate;

              auto func = acceptedModule->lookupSymbol<func::FuncOp>("foo");
              func.setName(inputFunctionName);
              func->removeAttr("llvm.emit_c_interface");
              func->setAttr("irsynth.raised", UnitAttr::get(&ctx));

              return failure();
            }

            // Print candidate.
            printCandidate(status, localCandidateStore, candidateStore,
                           newCandidate, options, module);
            return success();
          });
      if (failed(status)) {
        auto argIds = acceptedCandidate->getArgIds();
        return std::make_tuple(std::move(acceptedModule), argIds);
      }
    }

    candidateStore->merge(localCandidateStore);
  }

  if (options.printStats) {
    llvm::outs() << "\n";
    stats.dump();
  }

  return std::make_tuple(nullptr, std::vector<unsigned>());
}
