#include "PolyhedralAnalysis.h"

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

namespace {
struct PolyhedralInfoPass
    : public PassWrapper<PolyhedralInfoPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolyhedralInfoPass)

  PolyhedralInfoPass() = default;
  PolyhedralInfoPass(const PolyhedralInfoPass &) {}

  Option<bool> dump{*this, "dump", llvm::cl::desc("Dump all info"),
                    llvm::cl::init(false)};

  Option<bool> dot{*this, "dot", llvm::cl::desc("Dot graph"),
                   llvm::cl::init(false)};

  Option<bool> dotStmts{*this, "dot-stmts",
                        llvm::cl::desc("Dot graph of statement contents"),
                        llvm::cl::init(false)};

  StringRef getArgument() const final { return "polyhedral-info"; }
  StringRef getDescription() const final {
    return "Print polyhedral analysis info.";
  }
  void runOnOperation() override {

    Operation *op = getOperation();

    Scop scop(op);

    if (dump) {
      scop.dump(llvm::outs());
      llvm::outs() << "\n";
    }

    if (dot) {
      toDot(llvm::outs(), scop);
      llvm::outs() << "\n";
    }

    if (dotStmts) {
      toDotStmts(llvm::outs(), scop);
      llvm::outs() << "\n";
    }
  }
};
} // namespace

namespace mlir {
void registerPolyhedralInfoPass() { PassRegistration<PolyhedralInfoPass>(); }
} // namespace mlir

int main(int argc, char **argv) {
  registerPolyhedralInfoPass();

  DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
