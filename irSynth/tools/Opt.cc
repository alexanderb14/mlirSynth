#include "analysis/PolyhedralAnalysis.h"
#include "transforms/CopyModifiedMemrefsPass.h"
#include "transforms/LoopDistributionPass.h"
#include "transforms/LoopOutlinePass.h"
#include "transforms/MemrefMinifyPass.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);

  registerPolyhedralAnalysisPass();

  registerCopyModifiedMemrefsPass();
  registerLoopDistributionPass();
  registerLoopOutlinePass();
  registerMemrefMinifyPass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Synthesizer opt driver\n", registry, true));
}
