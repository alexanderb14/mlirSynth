#ifndef IRSYNTH_ATTRIBUTEGEN_H
#define IRSYNTH_ATTRIBUTEGEN_H

#include "enumeration/OpInfos.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"

std::vector<std::pair<mlir::Attribute, OpAndResType>>
genAttributes(mlir::OpBuilder &builder,
              mlir::Region::BlockArgListType &functionArgs,
              llvm::ArrayRef<int64_t> &targetShape, int maxRank);

std::vector<std::shared_ptr<mlir::Region>> genRegions(mlir::OpBuilder &builder);

#endif // IRSYNTH_ATTRIBUTEGEN_H
