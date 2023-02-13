#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <map>
#include <set>

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
  os << "Classes:\n";
  for (const auto &it : records.getClasses()) {
    os << it.second->getName() << "\n";
  }

  os << "\n";
  os << "Type definitions:\n";
  for (auto *record : getTypeDefinitions(records)) {
    os << record->getName() << "\n";
  }

  os << "\n";
  os << "Attr definitions:\n";
  for (auto *record : getAttrDefinitions(records)) {
    os << record->getName() << "\n";
  }

  os << "\n";
  for (auto *record : getOpDefinitions(records)) {
    os << "Op: " << record->getName() << "\n";
    auto tblgenOp = tblgen::Operator(record);

    // Operands
    os << "  Operands:\n";
    for (auto &operand : tblgenOp.getOperands()) {
      // os << "    " << operand.name << ": ";
      os << "    ";
      os << operand.constraint.getDefName() << "\n";
    }

    // Results
    os << "  Results:\n";
    for (auto &result : tblgenOp.getResults()) {
      // os << "    " << result.name << ": ";
      os << "    ";
      os << result.constraint.getDefName() << "\n";
    }
  }
}

void emitHdrIncludes(raw_ostream &os) {
  os << "#include <memory>\n";
  os << "#include <string>\n";
  os << "\n";
}

void emitSrcIncludes(raw_ostream &os) {
  os << "#include \"OpInfos.h\"\n";
  os << "\n";
  os << "#include <cassert>\n";
  os << "#include <memory>\n";
  os << "#include <string>\n";
  os << "\n";
}

void emitUsedTypesAsEnum(const RecordKeeper &records, raw_ostream &os) {
  std::set<std::string> usedTypes;
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = tblgen::Operator(record);
    for (auto &operand : tblgenOp.getOperands()) {
      usedTypes.insert(operand.constraint.getDefName().str());
    }
    for (auto &result : tblgenOp.getResults()) {
      usedTypes.insert(result.constraint.getDefName().str());
    }
  }

  os << "enum IOType {\n";
  unsigned size = usedTypes.size();
  for (auto &type : usedTypes) {
    os << "  " << type;
    if (--size > 0) {
      os << ",\n";
    } else {
      os << "\n";
    }
  }
  os << "};\n";
  os << "\n";
}

void emitAbstractOp(raw_ostream &os) {
  os << "class OpInfo {\n";
  os << "public:\n";
  os << "  virtual ~OpInfo() {}\n";
  os << "  virtual unsigned getNumOperands() const = 0;\n";
  os << "  virtual unsigned getNumResults() const = 0;\n";
  os << "  virtual IOType getOperandType(unsigned index) const = 0;\n";
  os << "  virtual IOType getResultType(unsigned index) const = 0;\n";
  os << "};\n";
  os << "using OpInfoPtr = std::unique_ptr<OpInfo>;\n";
  os << "\n";
}

void emitConstructorFnDecl(raw_ostream &os) {
  os << "OpInfoPtr createOpInfo(std::string name);\n";
}

std::string makeClangCompatible(const std::string &name) {
  std::string res = name;
  std::replace(res.begin(), res.end(), '.', '_');
  std::replace(res.begin(), res.end(), '-', '_');
  return res;
}

void emitConcreteOps(const RecordKeeper &records, raw_ostream &os) {
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = tblgen::Operator(record);

    std::string opName = makeClangCompatible(tblgenOp.getOperationName());
    os << "class " << opName << " : public OpInfo {\n";
    os << "public:\n";

    os << "  unsigned getNumOperands() const override {";
    os << " return " << tblgenOp.getNumOperands() << ";";
    os << " }\n";

    os << "  unsigned getNumResults() const override {";
    os << " return " << tblgenOp.getNumResults() << ";";
    os << " }\n";

    // Operands
    os << "  IOType getOperandType(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumOperands(); ++i) {
      auto &operand = tblgenOp.getOperand(i);
      os << "      case " << i << ": return " << operand.constraint.getDefName()
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid operand index\");\n";
    os << "  }\n";

    // Results
    os << "  IOType getResultType(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumResults(); ++i) {
      auto &result = tblgenOp.getResult(i);
      os << "      case " << i << ": return " << result.constraint.getDefName()
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid result index\");\n";
    os << "  }\n";

    os << "};\n";
    os << "\n";
  }
}

void emitConstructorFn(const RecordKeeper &records, raw_ostream &os) {
  os << "OpInfoPtr createOpInfo(std::string name) {\n";
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = tblgen::Operator(record);

    std::string opName = makeClangCompatible(tblgenOp.getOperationName());

    os << "  if (name == \"" << tblgenOp.getOperationName() << "\")\n";
    os << "    return std::make_unique<" << opName << ">();\n";
  }
  os << "  assert(false && \"Invalid op name\");\n";
  os << "}\n";
}

static bool emitOpInfoDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Getters for Operation Infos", os);
  emitHdrIncludes(os);

  emitUsedTypesAsEnum(recordKeeper, os);
  emitAbstractOp(os);
  emitConstructorFnDecl(os);

  return false;
}

static bool emitOpInfoDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Getters for Operation Infos", os);
  emitSrcIncludes(os);

  emitConcreteOps(recordKeeper, os);
  emitConstructorFn(recordKeeper, os);

  return false;
}

static mlir::GenRegistration
    genOpInfoDecls("gen-op-info-decls", "Generate op info declarations",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitOpInfoDecls(records, os);
                   });

static mlir::GenRegistration
    genOpInfoDefs("gen-op-info-defs", "Generate op info definitions",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    return emitOpInfoDefs(records, os);
                  });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
