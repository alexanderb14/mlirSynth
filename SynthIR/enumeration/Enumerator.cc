#include "Enumerator.h"

#include "CandidateTuples.h"
#include "Utils.h"
#include "enumeration/Candidate.h"
#include "execution/ArgUtils.h"

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
  reject_isNotAllDefsAreUsed,
  reject_isNotVerifiable,
  reject_hasNoArguments,
  reject_hasUnknownRankAndShape,
  reject_isNotCompilableToLLVM,
  reject_hasEmptyReturnShape,
  reject_isNotExecutable,
  reject_hashNotUnique,
  accept_candidate,
  accept_solution,
};

std::string processingStatusToStr(ProcessingStatus &status) {
  if (status == reject_hasTooManyOps)
    return "reject_hasTooManyOps";
  if (status == reject_isNotResultTypeInferrable)
    return "reject_isNotResultTypeInferrable";
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
  if (status == accept_candidate)
    return "accept_candidate";
  if (status == accept_solution)
    return "accept_solution";
  assert(false && "Processing Status not known");
}

void printCandidate(ProcessingStatus status, CandidateStorePtr &localCandidateStore,
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
    }
    else {
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

    if (status == accept_candidate) {
      statusStr = "\033[1;42m" + statusStr + "\033[0m";
    } else {
      statusStr = "\033[1;41m" + statusStr + "\033[0m";
    }
  }

  // Print the module.
  if ((status == accept_candidate && options.printValidCandidates) ||
      (!(status == accept_candidate) && options.printInvalidCandidates) ||
      options.printStatusNames) {
    llvm::outs() << statusStr << "\n";
    if (status > reject_isNotResultTypeInferrable)
      module->print(llvm::outs());
  }
}

void printStats(EnumerationStats &stats) {
  llvm::outs() << "\nEnumeration Stats"
               << "\n--------\n";
  llvm::outs() << "Number of enumerated candidates:             "
               << stats.numEnumerated << "\n";

  llvm::outs() << "Number of valid candidates:                  "
               << stats.numValid << "\n";
  llvm::outs() << "Percentage of valid candidates:              "
               << (stats.numValid * 100.0) / stats.numEnumerated << "%\n";

  llvm::outs() << "Number of executed candidates:               "
               << stats.numExecuted << "\n";
  llvm::outs() << "Percentage of executed candidates:           "
               << (stats.numExecuted * 100.0) / stats.numEnumerated << "%\n";

  llvm::outs() << "Number of ignored equivalent candidates:     "
               << stats.numEquivalent << "\n";
}

void printArray(double *arr, ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    llvm::outs() << arr[0] << "\n";
  } else if (shape.size() == 1) {
    for (int i = 0; i < shape[0]; i++) {
      llvm::outs() << arr[i] << " ";
    }
    llvm::outs() << "\n";
  } else if (shape.size() == 2) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        llvm::outs() << arr[i * shape[1] + j] << " ";
      }
      llvm::outs() << "\n";
    }
  } else {
    assert(false && "Unsupported shape");
  }
}

double hashArray(double *arr, ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return *arr;
  }
  if (shape.size() == 1) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      sum += arr[i];
    }
    return sum / shape[0] * 7.331;
  }
  if (shape.size() == 2) {
    double sum = 0;
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        sum += arr[i * shape[1] + j];
      }
    }
    return sum / (shape[0] * 1.337 + shape[1] * 0.337);
  }
  assert(false && "Unsupported shape");
}

bool areArraysEqual(double *arr1, double *arr2, ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return (floor(*arr1 * 1000) != floor(*arr2 * 1000));
  }
  if (shape.size() == 1) {
    for (int i = 0; i < shape[0]; i++) {
      if (floor(arr1[i] * 1000) != floor(arr2[i] * 1000)) {
        return false;
      }
    }
    return true;
  }
  if (shape.size() == 2) {
    for (int i = 0; i < shape[0]; i++) {
      for (int j = 0; j < shape[1]; j++) {
        if (floor(arr1[i * shape[1] + j] * 1000) !=
            floor(arr2[i * shape[1] + j] * 1000)) {
          return false;
        }
      }
    }
    return true;
  }
  assert(false && "Unsupported shape");
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
  for (auto &attr : getTensorAttributes(builder)) {
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
        EnumerationOptions &options, CandidateTuple operandCandidateTuple,
        CandidatePtr &newCandidate, OwningOpRef<ModuleOp> &module) {
  stats.numEnumerated++;

  // Create candidate.
  newCandidate.reset(new Candidate(operandCandidateTuple.operands));
  auto builder = OpBuilder(&ctx);

  // Set up operands.
  SmallVector<mlir::Value> operands =
      newCandidate->merge(ctx, operandCandidateTuple.operands);

  // Set up attributes.
  auto attrNames = getFilteredAttributeNames(opName);
  auto attrValues = operandCandidateTuple.attributes;
  assert(attrNames.size() == attrValues.size() &&
         "Attribute names and values must have the same size.");

  SmallVector<NamedAttribute> attributes = {};
  for (unsigned i = 0; i < attrNames.size(); i++) {
    StringAttr attrName = attrNames[i];
    mlir::Attribute value = attrValues[i];

    attributes.push_back(builder.getNamedAttr(attrName, value));
  }

  // Set up regions.
  SmallVector<std::unique_ptr<Region>> regions = {};
  for (auto &regionCandidate : operandCandidateTuple.regions) {
    std::unique_ptr<Region> region = std::make_unique<Region>();
    BlockAndValueMapping mapping;
    regionCandidate->cloneInto(region.get(), mapping);
    regions.push_back(std::move(region));
  }

  // Set up results types.
  // TODO: Parse number of results from the op definition.
  SmallVector<mlir::Type> resultTypes = {operands[0].getType()};

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
    return reject_isNotResultTypeInferrable;
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

  // if (!hasTargetShape(op, targetShape))
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
  if (options.ignoreEquivalentCandidates &&
      !candidateStore->addCandidateHash(hash)) {
    stats.numEquivalent++;
    return reject_hashNotUnique;
  }

  localCandidateStore->addCandidate(newCandidate, newCandidate->getNumOps());

  if (areArraysEqual(refOut, out, returnShape)) {
    llvm::outs() << "Found a match!\n";
    printArray(out, returnShape);
    module->dump();
    candidateStore->merge(localCandidateStore);
    printStats(stats);

    return accept_solution;
  }
  
  return accept_candidate;
}

bool enumerateCandidates(MLIRContext &ctx, IExecutorPtr executor,
                         func::FuncOp inputFunction,
                         CandidateStorePtr &candidateStore,
                         std::vector<RegisteredOperationName> &avaliableOps,
                         EnumerationOptions &options) {
  auto targetShape = getReturnShape(inputFunction);

  // Compile and run reference.
  // - Create argument vector.
  auto args = createArgs(inputFunction.getArguments());
  randomlyInitializeArgs(args);
  // printArgs(args);
  auto ret = getOwningMemRefForShape(targetShape);

  // - Run on argument vector gives the reference out.
  auto inputModuleRef = createModule(ctx, &inputFunction);
  auto inputModule = inputModuleRef.release();
  assert(succeeded(executor->lowerAffineToLLVMDialect(inputModule)));
  assert(succeeded(jitAndInvoke(inputModule, args, ret, false)));

  double *refOut = getReturnDataPtr(ret);
  printArray(refOut, targetShape);

  convertScalarToMemrefArgs(args);

  // Synthesize.
  // - Initialize candidate store with constant and argument candidates.
  initializeCandidates(ctx, candidateStore, inputFunction.getArguments());
  // - Print them.
  for (auto &candidate : candidateStore->getCandidates()) {
    auto module = createModule(ctx, candidate->getRegion());
    printCandidate(ProcessingStatus::accept_candidate, candidateStore, candidateStore,
                   candidate, options, module);
  }

  // - Enumerate candidates.
  EnumerationStats stats;
  for (int numOps = 0; numOps <= options.maxNumOps; numOps++) {
    if (options.printStats) {
      llvm::outs() << "\nLength: " << numOps << "\n";
      candidateStore->dumpSizes();
    }

    CandidateStorePtr localCandidateStore = std::make_shared<CandidateStore>();

    for (auto opName : avaliableOps) {
      auto operandCandidates = candidateStore->getCandidates(numOps);
      auto operandCandidateTuples =
          getOperandCandidateTuples(ctx, opName, operandCandidates);

      auto status = failableParallelForEach(
          &ctx, operandCandidateTuples, [&](auto &operandCandidateTuple) {
            CandidatePtr newCandidate;
            OwningOpRef<ModuleOp> module;

            ProcessingStatus status =
                process(ctx, stats, opName, executor, args, candidateStore,
                        localCandidateStore, refOut, options,
                        operandCandidateTuple, newCandidate, module);

            if (status == accept_solution)
              return failure();

            // Print candidate.
            printCandidate(status, localCandidateStore, candidateStore,
                           newCandidate, options, module);
            return success();
          });
      if (failed(status))
        return true;
    }

    candidateStore->merge(localCandidateStore);
  }

  llvm::outs() << "\n";
  printStats(stats);

  return false;
}
