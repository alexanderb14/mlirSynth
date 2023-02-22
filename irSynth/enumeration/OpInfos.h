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
  HLO_StaticShapeIntFpOrComplexTensor,
  HLO_StaticShapeIntOrFpTensor,
  HLO_StaticShapeTensor,
  HLO_StaticShapeTensorOrToken,
  HLO_Tensor,
  HLO_TensorOrToken,
  HLO_Token,
  HLO_Tuple,
  I32Tensor,
  Index,
  MHLO_AsyncBundle,
  Shape_WitnessType,
  anonymous_516,
  anonymous_541,
  anonymous_629,
  anonymous_638,
  anonymous_648,
  anonymous_677,
  anonymous_679,
  anonymous_700,
  anonymous_714,
  anonymous_716,
  anonymous_722,
  anonymous_728,
  anonymous_737,
  anonymous_744
};

enum AttrType {
  DefaultUnknownAttrType,
  ArrayAttr,
  BoolAttr,
  CHLO_ComparisonDirectionAttr,
  CHLO_ComparisonTypeAttr,
  ElementsAttr,
  F32Attr,
  FlatSymbolRefAttr,
  I32Attr,
  I64Attr,
  I64ElementsAttr,
  MHLO_ArrayOfLayoutAttr,
  MHLO_BoolElementsAttr,
  MHLO_ChannelHandle,
  MHLO_ComparisonDirectionAttr,
  MHLO_ComparisonTypeAttr,
  MHLO_ConvDimensionNumbers,
  MHLO_CustomCallApiVersionAttr,
  MHLO_CustomCallScheduleAttr,
  MHLO_DomainKindAttr,
  MHLO_DotDimensionNumbers,
  MHLO_FftTypeAttr,
  MHLO_FlatSymbolRefArrayAttr,
  MHLO_FusionKindAttr,
  MHLO_GatherDimensionNumbers,
  MHLO_PrecisionConfigAttr,
  MHLO_RngAlgorithmAttr,
  MHLO_RngDistributionAttr,
  MHLO_ScatterDimensionNumbers,
  MHLO_TransposeAttr,
  StrAttr,
  TypedAttrInterface,
  UnitAttr,
  anonymous_687,
  anonymous_694,
  anonymous_735
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
