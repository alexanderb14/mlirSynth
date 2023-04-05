/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Grammar (generated from tablegen)                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef IRSYNTH_GRAMMAR_H
#define IRSYNTH_GRAMMAR_H


#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <string>
#include <vector>

namespace grammar {
enum OpAndResType {
  HLO_AsyncBundle,
  HLO_ComplexTensor,
  HLO_DimensionTensor,
  HLO_Fp32Or64Tensor,
  HLO_FpOrComplexTensor,
  HLO_FpTensor,
  HLO_IntFpOrComplexTensor,
  HLO_IntOrFpTensor,
  HLO_IntTensor,
  HLO_PredIntOrFpTensor,
  HLO_PredOrIntTensor,
  HLO_PredTensor,
  HLO_QuantizedIntTensor,
  HLO_ScalarIntTensor,
  HLO_StaticShapeTensor,
  HLO_Tensor,
  HLO_TensorOrToken,
  HLO_TensorOrTokenOrTuple,
  HLO_Token,
  HLO_Tuple,
  I32Tensor,
  Index,
  Shape_WitnessType,
  anonymous_526,
  anonymous_632,
  anonymous_641,
  anonymous_651,
  anonymous_686,
  anonymous_688,
  anonymous_713,
  anonymous_726,
  anonymous_728,
  anonymous_734,
  anonymous_740,
  anonymous_747,
  anonymous_754
};

class AttributeGenerator {
public:
  AttributeGenerator(mlir::MLIRContext &ctx) : ctx(ctx) {}

  // AttrDef generators
  std::vector<mlir::Attribute> genChloComparisonDirectionAttr();
  std::vector<mlir::Attribute> genChloComparisonTypeAttr();
  std::vector<mlir::Attribute> genMhloArgResultAliasAttr();
  std::vector<mlir::Attribute> genMhloChannelHandleAttr();
  std::vector<mlir::Attribute> genMhloComparisonDirectionAttr();
  std::vector<mlir::Attribute> genMhloComparisonTypeAttr();
  std::vector<mlir::Attribute> genMhloConvDimensionNumbersAttr();
  std::vector<mlir::Attribute> genMhloDequantizeModeAttr();
  std::vector<mlir::Attribute> genMhloDomainKindAttr();
  std::vector<mlir::Attribute> genMhloDotDimensionNumbersAttr();
  std::vector<mlir::Attribute> genMhloFftTypeAttr();
  std::vector<mlir::Attribute> genMhloFusionKindAttr();
  std::vector<mlir::Attribute> genMhloGatherDimensionNumbersAttr();
  std::vector<mlir::Attribute> genMhloOutputOperandAliasAttr();
  std::vector<mlir::Attribute> genMhloPrecisionAttr();
  std::vector<mlir::Attribute> genMhloRngAlgorithmAttr();
  std::vector<mlir::Attribute> genMhloRngDistributionAttr();
  std::vector<mlir::Attribute> genMhloScatterDimensionNumbersAttr();
  std::vector<mlir::Attribute> genMhloTransposeAttr();
  std::vector<mlir::Attribute> genMhloTypeExtensionsAttr();

  // Attr generators
  std::vector<mlir::Attribute> genArrayAttr();
  std::vector<mlir::Attribute> genBoolAttr();
  std::vector<mlir::Attribute> genChannelHandleAttr();
  std::vector<mlir::Attribute> genComparisonTypeAttr();
  std::vector<mlir::Attribute> genCustomCallApiVersionAttr();
  std::vector<mlir::Attribute> genDenseElementsAttr();
  std::vector<mlir::Attribute> genDenseIntElementsAttr();
  std::vector<mlir::Attribute> genElementsAttr();
  std::vector<mlir::Attribute> genFlatSymbolRefAttr();
  std::vector<mlir::Attribute> genFloatAttr();
  std::vector<mlir::Attribute> genFusionKindAttr();
  std::vector<mlir::Attribute> genIntegerAttr();
  std::vector<mlir::Attribute> genStringAttr();
  std::vector<mlir::Attribute> genTypedAttr();
  std::vector<mlir::Attribute> genUnitAttr();

  // Types used in enums
  std::vector<::llvm::SmallVector<int64_t>> genLlvmSmallVectorint64t();
  std::vector<bool> genBool();
  std::vector<int64_t> genInt64t();

private:
  mlir::MLIRContext &ctx;
};
using AttributeGeneratorPtr = std::shared_ptr<AttributeGenerator>;

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
  virtual std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const = 0;
  virtual OpAndResType getResultType(unsigned index) const = 0;
};
using GrammarOpPtr = std::unique_ptr<GrammarOp>;
std::string opAndResTypeToString(OpAndResType type);
GrammarOpPtr createGrammarOp(std::string name);

} // namespace grammar
#endif // IRSYNTH_GRAMMAR_H
