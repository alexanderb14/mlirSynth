/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Getters for Operation Infos                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef IRSYNTH_OPINFOS_H
#define IRSYNTH_OPINFOS_H

#include <memory>
#include <string>

enum OpAndResType {
  DefaultUnknownOpAndResType,
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

enum AttrType {
  DefaultUnknownAttrType,
  ArrayAttr,
  BoolAttr,
  BoolElementsAttr,
  CHLO_ComparisonDirectionAttr,
  CHLO_ComparisonTypeAttr,
  ChannelHandle,
  ConvDimensionNumbers,
  DotDimensionNumbers,
  ElementsAttr,
  F32Attr,
  FlatSymbolRefAttr,
  GatherDimensionNumbers,
  HLO_ArrayOfLayoutAttr,
  HLO_ComparisonDirectionAttr,
  HLO_ComparisonTypeAttr,
  HLO_CustomCallApiVersionAttr,
  HLO_DomainKindAttr,
  HLO_FftTypeAttr,
  HLO_FlatSymbolRefArrayAttr,
  HLO_FusionKindAttr,
  HLO_PrecisionConfigAttr,
  HLO_RngAlgorithmAttr,
  HLO_RngDistributionAttr,
  HLO_TransposeAttr,
  I32Attr,
  I64Attr,
  I64ElementsAttr,
  ScatterDimensionNumbers,
  StrAttr,
  TypedAttrInterface,
  UnitAttr,
  anonymous_707
};

class OpInfo {
public:
  virtual ~OpInfo() {}
  virtual unsigned getNumOperands() const = 0;
  virtual unsigned getNumAttributes() const = 0;
  virtual unsigned getNumRegions() const = 0;
  virtual unsigned getNumResults() const = 0;
  virtual OpAndResType getOperandType(unsigned index) const = 0;
  virtual AttrType getAttributeType(unsigned index) const = 0;
  virtual std::string getAttributeName(unsigned index) const = 0;
  virtual OpAndResType getResultType(unsigned index) const = 0;
};
using OpInfoPtr = std::unique_ptr<OpInfo>;

std::string opAndResTypeToString(OpAndResType type);
std::string attrTypeToString(AttrType type);
OpInfoPtr createOpInfo(std::string name);

#endif // IRSYNTH_OPINFOS_H
