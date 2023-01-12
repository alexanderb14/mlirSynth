#ifndef IRSYNTH_LOWERING_H
#define IRSYNTH_LOWERING_H

#include "mlir/Pass/PassManager.h"

void addCHLOToLLVMPasses(std::shared_ptr<mlir::PassManager> pm);
void addAffineToLLVMPasses(std::shared_ptr<mlir::PassManager> pm);

#endif // IRSYNTH_EXECUTOR_H
