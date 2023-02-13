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

std::vector<std::string> getUsedTypes(const RecordKeeper &records) {
  std::set<std::string> types;
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = tblgen::Operator(record);
    for (auto &operand : tblgenOp.getOperands()) {
      types.insert(operand.constraint.getDefName().str());
    }
    for (auto &result : tblgenOp.getResults()) {
      types.insert(result.constraint.getDefName().str());
    }
  }

  // Add Unknown type as 1st element.
  std::vector<std::string> typesVec;
  typesVec.reserve(types.size());
for (auto &type : types) {
    typesVec.push_back(type);
  }
  typesVec.insert(typesVec.begin(), "DefaultUnknown");

  return typesVec;
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
  auto types = getUsedTypes(records);

  os << "enum IOType {\n";
  unsigned size = types.size();
  for (auto &type : types) {
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

void emitIOTypeToStringDecl(raw_ostream &os) {
  os << "std::string ioTypeToString(IOType type);\n";
}

void emitIOTypeToStringFn(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedTypes(records);

  os << "std::string IOTypeToString(IOType type) {\n";
  for (auto &type : types) {
    os << "  if (type == " << type << ") return \"" << type << "\";\n";
  }
  os << "  assert(false && \"Invalid IOType\");\n";
  os << "}\n";
  os << "\n";
}

void emitConstructorDecl(raw_ostream &os) {
  os << "OpInfoPtr createOpInfo(std::string name);\n";
  os << "\n";
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
  os << "\n";
}

void emitIncludeGuardStart(raw_ostream &os, const std::string &guard) {
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n";
  os << "\n";
}

void emitIncludeGuardEnd(raw_ostream &os, const std::string &guard) {
  os << "#endif // " << guard << "\n";
}

static bool emitOpInfoDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Getters for Operation Infos", os);
  emitIncludeGuardStart(os, "IRSYNTH_OPINFOS_H");
  emitHdrIncludes(os);

  emitUsedTypesAsEnum(recordKeeper, os);
  emitAbstractOp(os);
  emitIOTypeToStringDecl(os);
  emitConstructorDecl(os);
  emitIncludeGuardEnd(os, "IRSYNTH_OPINFOS_H");

  return false;
}

static bool emitOpInfoDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Getters for Operation Infos", os);
  emitSrcIncludes(os);

  emitConcreteOps(recordKeeper, os);
  emitIOTypeToStringFn(recordKeeper, os);
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
