#ifndef IRSYNTH_ATTRIBUTEGEN_H
#define IRSYNTH_ATTRIBUTEGEN_H

#include "enumeration/Candidate.h"
#include "enumeration/Grammar.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"

// Initial candidate generators
std::vector<CandidatePtr>
genInitialCandidates(mlir::MLIRContext &ctx,
                     mlir::Region::BlockArgListType functionArgs,
                     llvm::ArrayRef<int64_t> targetShape);

// Attribute generators
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
std::vector<std::shared_ptr<mlir::Region>> genRegions(mlir::MLIRContext &ctx);

#endif // IRSYNTH_ATTRIBUTEGEN_H
