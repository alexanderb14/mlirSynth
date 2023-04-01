#ifndef IRSYNTH_ATTRIBUTEGEN_H
#define IRSYNTH_ATTRIBUTEGEN_H

#include "enumeration/Grammar.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"

std::vector<std::pair<mlir::Attribute, grammar::OpAndResType>>
genAttributes(mlir::MLIRContext &ctx,
              mlir::Region::BlockArgListType &functionArgs,
              llvm::ArrayRef<int64_t> &targetShape, int maxRank);

std::vector<std::shared_ptr<mlir::Region>> genRegions(mlir::MLIRContext &ctx);

#endif // IRSYNTH_ATTRIBUTEGEN_H
