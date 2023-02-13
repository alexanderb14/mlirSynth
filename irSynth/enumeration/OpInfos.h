/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Getters for Operation Infos                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include <memory>
#include <string>

enum IOType {
  DefaultUnknown,
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
  anonymous_594,
  anonymous_603,
  anonymous_614,
  anonymous_651,
  anonymous_653,
  anonymous_678,
  anonymous_691,
  anonymous_693,
  anonymous_699,
  anonymous_705,
  anonymous_712,
  anonymous_719
};

class OpInfo {
public:
  virtual ~OpInfo() {}
  virtual unsigned getNumOperands() const = 0;
  virtual unsigned getNumResults() const = 0;
  virtual IOType getOperandType(unsigned index) const = 0;
  virtual IOType getResultType(unsigned index) const = 0;
};
using OpInfoPtr = std::unique_ptr<OpInfo>;

std::string ioTypeToString(IOType type);
OpInfoPtr createOpInfo(std::string name);
