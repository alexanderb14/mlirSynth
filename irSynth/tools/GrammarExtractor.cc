#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

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

std::vector<std::string> getUsedOpAndResTypes(const RecordKeeper &records) {
  std::set<std::string> types;
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);
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
  typesVec.insert(typesVec.begin(), "DefaultUnknownOpAndResType");

  return typesVec;
}

std::vector<std::string> getUsedAttrTypes(const RecordKeeper &records) {
  std::set<std::string> types;
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);
    for (auto &attr : tblgenOp.getAttributes()) {
      types.insert(attr.attr.getDefName().str());
    }
  }

  // Add Unknown type as 1st element.
  std::vector<std::string> typesVec;
  typesVec.reserve(types.size());
  for (auto &type : types) {
    typesVec.push_back(type);
  }
  typesVec.insert(typesVec.begin(), "DefaultUnknownAttrType");

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
    auto tblgenOp = Operator(record);

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
  os << R"(
#include "mlir/IR/Attributes.h"

#include <memory>
#include <string>
)";
}

void emitSrcIncludes(raw_ostream &os) {
  os << R"(
#include "Grammar.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TensorEncoding.h"
#include "stablehlo/dialect/Base.h"

// Include order below matters.
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_enums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_attrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_typedefs.h.inc"

// Include order matters
#include "stablehlo/dialect/ChloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/ChloAttrs.h.inc"

#include <cassert>
#include <memory>
#include <string>
)";
}

void emitUsedOpAndResTypesAsEnum(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedOpAndResTypes(records);

  os << "enum OpAndResType {\n";
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

void emitUsedAttrTypesAsEnum(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedAttrTypes(records);

  os << "enum AttrType {\n";
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

void emitUsedAttrTypeValues(const RecordKeeper &records, raw_ostream &os) {
  auto enumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");
  for (auto *enumDefRecord : enumDefs) {
    EnumAttr enumAttr(enumDefRecord);
    os << enumAttr.getEnumClassName() << "\n";
    for (auto enumerant : enumAttr.getAllCases()) {
      os << "  " << enumerant.getSymbol() << "\n";
    }
  }
  os << "\n";

  auto attrDefs = records.getAllDerivedDefinitionsIfDefined("AttrDef");
  for (auto *attrDefRecord : attrDefs) {
    AttrOrTypeDef attrDef(attrDefRecord);
    os << attrDef.getCppClassName() << "\n";

    for (auto param : attrDef.getParameters()) {
      os << "  " << param.getName();
      os << " : ";

      std::string paramType = param.getCppType().str();
      if (auto *paramDefInit = dyn_cast<llvm::DefInit>(param.getDef())) {
        auto *rec = paramDefInit->getDef();
        if (!rec->isSubClassOf("EnumParameter"))
          paramType = rec->getName();
      }

      // Exctract all what is after the rightmost ::
      auto pos = paramType.rfind("::");
      if (pos != std::string::npos)
        paramType = paramType.substr(pos + 2);
      os << paramType << "\n";
    }
  }
}

void emitAbstractOp(raw_ostream &os) {
  os << R"(
class GrammarOp {
public:
  virtual ~GrammarOp() {}
  virtual unsigned getNumOperands() const = 0;
  virtual unsigned getNumAttributes() const = 0;
  virtual unsigned getNumRegions() const = 0;
  virtual unsigned getNumResults() const = 0;
  virtual OpAndResType getOperandType(unsigned index) const = 0;
  virtual mlir::Attribute getAttributeType(unsigned index) const = 0;
  virtual std::string getAttributeName(unsigned index) const = 0;
  virtual OpAndResType getResultType(unsigned index) const = 0;
};
using GrammarOpPtr = std::unique_ptr<GrammarOp>;
)";
}

std::string makeClangCompatible(const std::string &name) {
  std::string res = name;
  std::replace(res.begin(), res.end(), '.', '_');
  std::replace(res.begin(), res.end(), '-', '_');
  return res;
}

void emitConcreteOps(const RecordKeeper &records, raw_ostream &os) {
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);

    std::string opName = makeClangCompatible(tblgenOp.getOperationName());
    os << "class " << opName << " : public GrammarOp {\n";
    os << "public:\n";

    os << "  unsigned getNumOperands() const override {";
    os << " return " << tblgenOp.getNumOperands() << ";";
    os << " }\n";

    os << "  unsigned getNumAttributes() const override {";
    os << " return " << tblgenOp.getNumAttributes() << ";";
    os << " }\n";

    os << "  unsigned getNumRegions() const override {";
    os << " return " << tblgenOp.getNumRegions() << ";";
    os << " }\n";

    os << "  unsigned getNumResults() const override {";
    os << " return " << tblgenOp.getNumResults() << ";";
    os << " }\n";

    // Operands
    os << "  OpAndResType getOperandType(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumOperands(); ++i) {
      auto &operand = tblgenOp.getOperand(i);
      os << "      case " << i << ": return " << operand.constraint.getDefName()
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid operand index\");\n";
    os << "  }\n";

    // Attributes
    os << "  mlir::Attribute getAttributeType(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);
      auto attrName = attr.attr.getDefName();
      auto attrType = attr.attr.getReturnType();
      os << "      case " << i << ": return " << attr.attr.getStorageType() << "()"
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid attribute index\");\n";
    os << "  }\n";

    os << "  std::string getAttributeName(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);
      os << "      case " << i << ": return \"" << attr.name.str()
         << "\";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid attribute index\");\n";
    os << "  }\n";

    // Results
    os << "  OpAndResType getResultType(unsigned index) const override {\n";
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

void emitOpAndResTypeToStringDecl(raw_ostream &os) {
  os << "std::string opAndResTypeToString(OpAndResType type);\n";
}

void emitOpAndResTypeToStringFn(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedOpAndResTypes(records);

  os << "std::string opAndResTypeToString(OpAndResType type) {\n";
  for (auto &type : types) {
    os << "  if (type == " << type << ") return \"" << type << "\";\n";
  }
  os << "  assert(false && \"Invalid OpAndResType\");\n";
  os << "}\n";
  os << "\n";
}

void emitAttrTypeToStringDecl(raw_ostream &os) {
  os << "std::string attrTypeToString(AttrType type);\n";
}

void emitAttrTypeToStringFn(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedAttrTypes(records);

  os << "std::string attrTypeToString(AttrType type) {\n";
  for (auto &type : types) {
    os << "  if (type == " << type << ") return \"" << type << "\";\n";
  }
  os << "  assert(false && \"Invalid AttrType\");\n";
  os << "}\n";
  os << "\n";
}

void emitConstructorDecl(raw_ostream &os) {
  os << "GrammarOpPtr createGrammarOp(std::string name);\n";
  os << "\n";
}

void emitConstructorFn(const RecordKeeper &records, raw_ostream &os) {
  os << "GrammarOpPtr createGrammarOp(std::string name) {\n";
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);

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

void emitNamespaceStart(raw_ostream &os, const std::string &ns) {
  os << "namespace " << ns << " {\n";
}

void emitNamespaceEnd(raw_ostream &os, const std::string &ns) {
  os << "} // namespace " << ns << "\n";
}

static bool emitGrammarOpDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Grammar (generated from tablegen)", os);
  emitIncludeGuardStart(os, "IRSYNTH_GRAMMAR_H");
  emitHdrIncludes(os);

  emitNamespaceStart(os, "grammar");
  emitUsedOpAndResTypesAsEnum(recordKeeper, os);
  //emitUsedAttrTypesAsEnum(recordKeeper, os);
  emitAbstractOp(os);
  emitOpAndResTypeToStringDecl(os);
  //emitAttrTypeToStringDecl(os);
  emitConstructorDecl(os);
  emitNamespaceEnd(os, "grammar");
  emitIncludeGuardEnd(os, "IRSYNTH_GRAMMAR_H");
  //emitUsedAttrTypeValues(recordKeeper, os);

  return false;
}

static bool emitGrammarOpDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Grammar (generated from tablegen)", os);
  emitSrcIncludes(os);

  emitNamespaceStart(os, "grammar");
  emitConcreteOps(recordKeeper, os);
  emitOpAndResTypeToStringFn(recordKeeper, os);
  //emitAttrTypeToStringFn(recordKeeper, os);
  emitConstructorFn(recordKeeper, os);
  emitNamespaceEnd(os, "grammar");

  return false;
}

static mlir::GenRegistration
    genGrammarDecls("gen-grammar-decls", "Generate grammar declarations",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitGrammarOpDecls(records, os);
                   });

static mlir::GenRegistration
    genGrammarDefs("gen-grammar-defs", "Generate grammar definitions",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    return emitGrammarOpDefs(records, os);
                  });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
