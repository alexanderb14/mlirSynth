#include "Synthesizer.h"

#include "Common.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"
#include "synthesis/Candidate.h"
#include "synthesis/CartesianProduct.h"
#include "synthesis/Generators.h"
#include "synthesis/Grammar.h"
#include "synthesis/ProcessingStatus.h"
#include "synthesis/Stats.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"

#include <chrono>
#include <cstdint>
#include <math.h>
#include <memory>
#include <optional>
#include <variant>

using namespace llvm;
using namespace mlir;

unsigned maxShapeRank = 4;

void printCandidate(ProcessingStatus status,
                    CandidateStorePtr &localCandidateStore,
                    CandidateStorePtr &candidateStore,
                    SynthesisOptions &options, SynthesisResultPtr &result) {
  // If there is nothing to print, return early.
  if (!(options.printStatusNames || options.printStatusTiles ||
        options.printValidCandidates || options.printInvalidCandidates)) {
    return;
  }

  // Build and print the status string.
  int candidateId;
  if (result) {
    candidateId = localCandidateStore->getCandidateId(result->candidate);
  } else {
    candidateId = -1;
  }

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
      for (auto &pred : result->candidate->getPredecessors()) {
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
    if (result->module && status > reject_hasUnsupportedShapeRank) {
      result->module->print(llvm::outs());
    }
  }
}

void prepareInputFunction(func::FuncOp &inputFunction) {
  inputFunction->setAttr("llvm.emit_c_interface",
                         UnitAttr::get(inputFunction->getContext()));
  inputFunction.setName("foo");
}

void finalizeFunction(func::FuncOp func, std::string &funcName) {
  func.setName(funcName);
  func->removeAttr("llvm.emit_c_interface");
  func->setAttr("irsynth.raised", UnitAttr::get(func->getContext()));
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

LogicalResult verifyOp(Operation *op, RegisteredOperationName &opName) {
  if (failed(opName.verifyTraits(op))) {
    return failure();
  }
  // bool verifyTraitsOnly = false;
  // if (op->getName().getStringRef().str() == "stablehlo.dot") {
  //   verifyTraitsOnly = true;
  // }

  // if (verifyTraitsOnly) {
  //   if (failed(opName.verifyTraits(op))) {
  //     return failure();
  //   }
  // } else {
  //   if (failed(opName.verifyInvariants(op))) {
  //     return failure();
  //   }
  // }
  return success();
}

LogicalResult inferResultTypes(MLIRContext &ctx, Operation *op) {
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

bool verifyDefsAreUsed(Block *block) {
  mlir::DenseMap<mlir::Value, bool> values;

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

ProcessingStatus process(MLIRContext &ctx, SynthesisStats &stats,
                         RegisteredOperationName &opName,
                         grammar::GrammarOpPtr &opInfo, IExecutorPtr &executor,
                         std::vector<ReturnAndArgType> &args,
                         CandidateStorePtr &candidateStore,
                         CandidateStorePtr &localCandidateStore, double *refOut,
                         SynthesisOptions &options, ArgTuple operandArgTuple,
                         SynthesisResultPtr &synthesisResult,
                         ArrayRef<int64_t> &targetShape) {
  stats.numSynthesized++;

  // Create candidate.
  CandidatePtr newCandidate = std::make_shared<Candidate>(
      operandArgTuple.operands, grammar::OpAndResType::HLO_Tensor);
  auto builder = OpBuilder(&ctx);

  synthesisResult = std::make_shared<SynthesisResult>();
  synthesisResult->candidate = newCandidate;

  // Set up operands.
  SmallVector<mlir::Value> operands =
      newCandidate->merge(ctx, operandArgTuple.operands);

  // Set up attributes.
  SmallVector<NamedAttribute> attributes = {};
  for (unsigned i = 0; i < opInfo->getNumAttributes(); i++) {
    if (!opInfo->isAttributeRequired(i))
      continue;

    std::string attrName = opInfo->getAttributeName(i);
    mlir::Attribute value = operandArgTuple.attributes[i];
    attributes.push_back(builder.getNamedAttr(attrName, value));
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
  Operation *op;
  if (opName.getIdentifier() == "linalg.matmul") {
    ValueRange ins = {operands[0], operands[1]};
    ValueRange outs = {operands[2]};
    op = builder.create<linalg::MatmulOp>(UnknownLoc::get(&ctx), ins, outs);
  } else if (opName.getIdentifier() == "linalg.matvec") {
    ValueRange ins = {operands[0], operands[1]};
    ValueRange outs = {operands[2]};
    op = builder.create<linalg::MatvecOp>(UnknownLoc::get(&ctx), ins, outs);
  } else {
    op = builder.create(UnknownLoc::get(&ctx), opName.getIdentifier(), operands,
                        resultTypes, attributes, {}, regions);
  }

  newCandidate->addOperation(ctx, op);

  // Check length.
  if (newCandidate->getNumOps() > options.maxNumOps) {
    return reject_hasTooManyOps;
  }

  // createModule(ctx, newCandidate->getRegion())->print(llvm::outs());

  // Infer the operation result type.
  if (failed(verifyOp(op, opName))) {
    return reject_isNotVerifiable;
  }

  if (!options.skipTypeInference) {
    if (failed(inferResultTypes(ctx, op))) {
      if (options.printInvalidCandidates)
        createModule(ctx, newCandidate->getRegion())->print(llvm::outs());
      return reject_isNotResultTypeInferrable;
    }
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
  OwningOpRef<ModuleOp> module = createModule(ctx, newCandidate->getRegion());

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
  if (options.withCopyArgs)
    argsCand = copyArgs(argsCand);
  auto returnShapeCand = getReturnShape(func);
  auto retCand = getOwningMemRefForShape(returnShapeCand);

  // Compile and run.
  if (failed(jitAndInvoke(moduleCopy, argsCand, retCand)))
    return reject_isNotExecutable;
  stats.numExecuted++;

  double *out = getReturnDataPtr(retCand);
  // printArray(out, returnShape, llvm::outs());
  if (options.printArgsAndResults)
    printArgsAndResultsInPython(args, out, targetShape);

  // Hash and add to store if hash doesn't exist yet.
  double hash = hashArray(out, returnShape);
  newCandidate->setHash(hash);
  // llvm::outs() << "Hash: " << hash << "\n";
  if (options.ignoreEquivalentCandidates &&
      !candidateStore->addCandidateHash(hash)) {
    stats.numIgnored++;
    return reject_hashNotUnique;
  }

  localCandidateStore->addCandidate(newCandidate);

  if (returnShape == targetShape) {
    if (areArraysEqual(refOut, out, returnShape)) {
      LLVM_DEBUG(llvm::dbgs() << "Found a match!\n");
      LLVM_DEBUG(module->print(llvm::dbgs()));
      // printArray(out, returnShape, llvm::outs());

      candidateStore->merge(localCandidateStore);
      stats.numOps = newCandidate->getNumOps();

      synthesisResult->module = module.release();
      return accept_as_solution;
    }
  }

  synthesisResult->module = module.release();
  return accept_as_candidate;
}

SynthesisResultPtr
synthesizeCandidates(MLIRContext &ctx, IExecutorPtr executor,
                    func::FuncOp inputFunction,
                    InitialCandidateGeneratorPtr initialCandidateGen,
                    CandidateStorePtr &candidateStore,
                    std::vector<RegisteredOperationName> &avaliableOps,
                    SynthesisOptions &options, SynthesisStats &stats) {
  auto inputFunctionName = inputFunction.getName().str();
  auto inputFunctionArgs = inputFunction.getArguments();
  auto targetShape = getReturnShape(inputFunction);
  prepareInputFunction(inputFunction);

  // Compile and run reference.
  // - Create argument vector.
  auto args = createArgs(inputFunction);
  randomlyInitializeArgs(inputFunction, args);
  auto ret = getOwningMemRefForShape(targetShape);

  // - Run on argument vector gives the reference out.
  auto inputModuleRef = createModule(ctx, &inputFunction);
  auto inputModule = inputModuleRef.release();
  assert(succeeded(executor->lowerAffineToLLVMDialect(inputModule)));
  assert(succeeded(jitAndInvoke(inputModule, args, ret)));

  double *refOut = getReturnDataPtr(ret);
  if (options.printArgsAndResults)
    printArgsAndResultsInPython(args, refOut, targetShape);

  convertScalarToMemrefArgs(args);

  // Synthesize.
  // - Initialize candidate store.
  auto candidates = initialCandidateGen->gen(inputFunctionArgs, targetShape);
  for (auto &candidate : candidates)
    candidateStore->addCandidate(candidate);

  // - Print them.
  for (auto &candidate : candidateStore->getCandidates()) {
    auto module = createModule(ctx, candidate->getRegion());

    SynthesisResultPtr synthesisResult =
        std::make_shared<SynthesisResult>();
    synthesisResult->candidate = candidate;
    synthesisResult->module = module.release();
    printCandidate(ProcessingStatus::accept_as_candidate, candidateStore,
                   candidateStore, options, synthesisResult);
  }

  CartesianProduct cartesianProduct(options.maxNumOps);

  // Get the current time.
  auto startTime = std::chrono::high_resolution_clock::now();

  SynthesisResultPtr result;

  // - Synthesize candidates.
  for (int numOps = 0; numOps <= options.maxNumOps; numOps++) {
    CandidateStorePtr localCandidateStore = std::make_shared<CandidateStore>();

    for (auto opName : avaliableOps) {
      // Build cartesian product of candidates.
      std::vector<std::vector<CandidatePtr>> operands;
      std::vector<std::vector<mlir::Attribute>> attributes;
      std::vector<std::vector<RegionPtr>> regions;

      // - Operands.
      auto opInfo = grammar::createGrammarOp(opName.getStringRef().str());
      for (unsigned i = 0; i < opInfo->getNumOperands(); i++) {
        std::vector<CandidatePtr> operandCandidates;
        if (options.ignoreTypes) {
          operandCandidates = candidateStore->getCandidates(numOps);
        } else {
          operandCandidates =
              candidateStore->getCandidates(numOps, opInfo->getOperandType(i));
        }
        operands.push_back(operandCandidates);
      }

      // - Attributes.
      auto attrGen = std::make_shared<AttributeGenerator>(
          ctx, inputFunctionArgs, targetShape);
      attributes = opInfo->genAttributes(attrGen);

      // - Regions.
      auto regionsGenereated = genRegions(ctx);
      for (unsigned i = 0; i < opInfo->getNumRegions(); i++) {
        regions.push_back(regionsGenereated);
      }

      auto operandArgTuples =
          cartesianProduct.generate(operands, attributes, regions);

      // Synthesize cartesian product.
      auto status = failableParallelForEach(
          &ctx, operandArgTuples, [&](auto &operandArgTuple) {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(
                                   endTime - startTime)
                                   .count();
            if (options.timeoutPerFunction &&
                elapsedTime > options.timeoutPerFunction)
              return failure();

            SynthesisResultPtr synthesisResult;
            SynthesisStats synthesisStats;

            ProcessingStatus status =
                process(ctx, synthesisStats, opName, opInfo, executor, args,
                        candidateStore, localCandidateStore, refOut, options,
                        operandArgTuple, synthesisResult, targetShape);
            synthesisStats.addProcessingStatus(status);
            stats.merge(synthesisStats);

            // Print candidate.
            printCandidate(status, localCandidateStore, candidateStore, options,
                           synthesisResult);

            if (status == accept_as_solution) {
              result = synthesisResult;
              finalizeFunction(
                  result->module->lookupSymbol<func::FuncOp>("foo"),
                  inputFunctionName);

              return failure();
            }

            return success();
          });
      if (failed(status)) {
        if (result)
          return result;
        return result;
      }
    }

    candidateStore->merge(localCandidateStore);
  }

  return result;
}
