#ifndef IRSYNTH_ATTRIBUTEGEN_H
#define IRSYNTH_ATTRIBUTEGEN_H

#include "synthesis/Candidate.h"
#include "synthesis/Grammar.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"

// Attribute generators
// -----------------------------------------------------------------------------
std::vector<std::pair<mlir::Attribute, grammar::OpAndResType>>
genAttributes(mlir::MLIRContext &ctx,
              mlir::Region::BlockArgListType &functionArgs,
              llvm::ArrayRef<int64_t> &targetShape);

class AttributeGenerator : public grammar::AttributeGeneratorBase {
public:
  AttributeGenerator(mlir::MLIRContext &ctx,
                     mlir::Region::BlockArgListType &functionArgs,
                     llvm::ArrayRef<int64_t> &targetShape)
      : grammar::AttributeGeneratorBase(ctx), functionArgs(functionArgs),
        targetShape(targetShape) {}

  std::vector<mlir::Attribute> genDenseIntElementsAttr() override;
  std::vector<::llvm::SmallVector<int64_t>> genLlvmSmallVectorint64t() override;

private:
  mlir::Region::BlockArgListType &functionArgs;
  llvm::ArrayRef<int64_t> &targetShape;
};
using AttributeGeneratorPtr = std::shared_ptr<AttributeGenerator>;

// Region generators
// -----------------------------------------------------------------------------
std::vector<std::shared_ptr<mlir::Region>> genRegions(mlir::MLIRContext &ctx);

// Initial candidate generators
// -----------------------------------------------------------------------------
class InitialCandidateGenerator {
public:
  InitialCandidateGenerator(mlir::MLIRContext &ctx) : ctx(ctx) {}
  virtual ~InitialCandidateGenerator() = default;

  virtual std::vector<CandidatePtr>
  gen(mlir::Region::BlockArgListType functionArgs,
      llvm::ArrayRef<int64_t> targetShape) = 0;

protected:
  mlir::MLIRContext &ctx;
};
using InitialCandidateGeneratorPtr = std::shared_ptr<InitialCandidateGenerator>;

class HLOInitialCandidateGenerator : public InitialCandidateGenerator {
public:
  HLOInitialCandidateGenerator(mlir::MLIRContext &ctx)
      : InitialCandidateGenerator(ctx) {}

  std::vector<CandidatePtr> gen(mlir::Region::BlockArgListType functionArgs,
                                llvm::ArrayRef<int64_t> targetShape) override;
};
using HLOInitialCandidateGeneratorPtr =
    std::shared_ptr<HLOInitialCandidateGenerator>;

class LinalgInitialCandidateGenerator : public InitialCandidateGenerator {
public:
  LinalgInitialCandidateGenerator(mlir::MLIRContext &ctx)
      : InitialCandidateGenerator(ctx) {}

  std::vector<CandidatePtr> gen(mlir::Region::BlockArgListType functionArgs,
                                llvm::ArrayRef<int64_t> targetShape) override;
};

#endif // IRSYNTH_ATTRIBUTEGEN_H
