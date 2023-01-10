#ifndef TOOLS_SYNTHESIZER_LOWERING_H
#define TOOLS_SYNTHESIZER_LOWERING_H

#include "mlir/Pass/PassManager.h"

void addCHLOToLLVMPasses(std::shared_ptr<mlir::PassManager> pm);
void addAffineToLLVMPasses(std::shared_ptr<mlir::PassManager> pm);

#endif // TOOLS_SYNTHESIZER_EXECUTOR_H
