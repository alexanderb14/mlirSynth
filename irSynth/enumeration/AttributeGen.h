#ifndef IRSYNTH_ATTRIBUTEGEN_H
#define IRSYNTH_ATTRIBUTEGEN_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Region.h"

std::vector<mlir::Attribute>
getTensorAttributes(mlir::OpBuilder &builder,
                    mlir::Region::BlockArgListType &functionArgs, int maxRank);

#endif // IRSYNTH_ATTRIBUTEGEN_H
