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
  virtual std::vector<mlir::Attribute> genAttributes(mlir::MLIRContext &ctx) const = 0;
  virtual OpAndResType getResultType(unsigned index) const = 0;
};
using GrammarOpPtr = std::unique_ptr<GrammarOp>;

class AttributeGenerator {
public:
  std::vector<mlir::Attribute> genMhloArgResultAliasAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genChloComparisonDirectionAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genChloComparisonTypeAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloChannelHandleAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloConvDimensionNumbersAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloDotDimensionNumbersAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloGatherDimensionNumbersAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloComparisonDirectionAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloComparisonTypeAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloDequantizeModeAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloDomainKindAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloFftTypeAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloFusionKindAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloPrecisionAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloRngAlgorithmAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloRngDistributionAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloTransposeAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloOutputOperandAliasAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloScatterDimensionNumbersAttr(mlir::MLIRContext &ctx);
  std::vector<mlir::Attribute> genMhloTypeExtensionsAttr(mlir::MLIRContext &ctx);
};

std::string opAndResTypeToString(OpAndResType type);
GrammarOpPtr createGrammarOp(std::string name);

} // namespace grammar
#endif // IRSYNTH_GRAMMAR_H
