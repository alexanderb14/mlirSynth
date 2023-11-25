#ifndef IRSYNTH_LOWERING_H
#define IRSYNTH_LOWERING_H

#include "mlir/Pass/PassManager.h"

namespace HLO {
void addCHLOToAffinePasses(std::shared_ptr<mlir::PassManager> pm);
void addAffineToLLVMPasses(std::shared_ptr<mlir::PassManager> pm);
} // namespace HLO

namespace Polygeist {
void addAffineToLLVMPasses(std::shared_ptr<mlir::PassManager> pm);
}

#endif // IRSYNTH_EXECUTOR_H
