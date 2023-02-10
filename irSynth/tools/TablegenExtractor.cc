#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace llvm;
using namespace mlir;

std::vector<Record *> getOpDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("Op"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("Op");
}

std::vector<Record *> getTypeDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("TypeDef"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("TypeDef");
}

std::vector<Record *> getAttrDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("AttrDef"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("AttrDef");
}

void printDefinitions(RecordKeeper &records, raw_ostream &os) {
  llvm::outs() << "Classes:\n";
  for (const auto & it : records.getClasses()) {
    llvm::outs() << it.second->getName() << "\n";
  }

  llvm::outs() << "\n";
  llvm::outs() << "Type definitions:\n";
  for (auto *record : getTypeDefinitions(records)) {
    llvm::outs() << record->getName() << "\n";
  }

  llvm::outs() << "\n";
  llvm::outs() << "Attr definitions:\n";
  for (auto *record : getAttrDefinitions(records)) {
    llvm::outs() << record->getName() << "\n";
  }

  llvm::outs() << "\n";
  for (auto *record : getOpDefinitions(records)) {
    llvm::outs() << "Op: " << record->getName() << "\n";
    auto tblgenOp = tblgen::Operator(record);

    // Operands
    llvm::outs() << "  Operands:\n";
    for (auto &operand : tblgenOp.getOperands()) {
      //llvm::outs() << "    " << operand.name << ": ";
      llvm::outs() << "    ";
      llvm::outs() << operand.constraint.getDefName() << "\n";
    }

    // Results
    llvm::outs() << "  Results:\n";
    for (auto &result : tblgenOp.getResults()) {
      //llvm::outs() << "    " << result.name << ": ";
      llvm::outs() << "    ";
      llvm::outs() << result.constraint.getDefName() << "\n";
    }
  }
}

bool extractTypes(raw_ostream &os, RecordKeeper &records) {
  printDefinitions(records, llvm::outs());

  return true;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  return TableGenMain(argv[0], &extractTypes);
}
