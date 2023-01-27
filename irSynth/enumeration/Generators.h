#ifndef IRSYNTH_ATTRIBUTEGEN_H
#define IRSYNTH_ATTRIBUTEGEN_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"

std::vector<mlir::Attribute>
genAttributes(mlir::OpBuilder &builder,
              mlir::Region::BlockArgListType &functionArgs, int maxRank);

std::vector<std::shared_ptr<mlir::Region>> genRegions(mlir::OpBuilder &builder);

#endif // IRSYNTH_ATTRIBUTEGEN_H
