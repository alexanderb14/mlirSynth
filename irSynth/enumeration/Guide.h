#ifndef IRSYNTH_ENUMERATION_GUIDE_H
#define IRSYNTH_ENUMERATION_GUIDE_H

#include "mlir/IR/Operation.h"

std::vector<std::string> predictOps(std::vector<std::string> &supportedOps, mlir::Operation *op);

#endif // IRSYNTH_ENUMERATION_GUIDE_H
