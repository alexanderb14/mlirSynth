#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Exporter.h"
#include "Scop.h"

using namespace llvm;
using namespace mlir;

namespace {
struct AffineInfoPass
    : public PassWrapper<AffineInfoPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineInfoPass)

  AffineInfoPass() = default;
  AffineInfoPass(const AffineInfoPass &) {}

  Option<bool> dump{*this, "dump", llvm::cl::desc("Dump all info"),
                    llvm::cl::init(false)};

  Option<bool> dot{*this, "dot", llvm::cl::desc("Dot graph"),
                   llvm::cl::init(false)};

  Option<bool> dotStmts{*this, "dot-stmts",
                        llvm::cl::desc("Dot graph of statement contents"),
                        llvm::cl::init(false)};

  StringRef getArgument() const final { return "affine-info"; }
  StringRef getDescription() const final { return "Print affine information."; }
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
void registerAffineInfoPass() { PassRegistration<AffineInfoPass>(); }
} // namespace mlir

int main(int argc, char **argv) {
  registerAffineInfoPass();

  DialectRegistry registry;
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
