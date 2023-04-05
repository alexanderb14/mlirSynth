/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Grammar (generated from tablegen)                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


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

namespace grammar {
std::string opAndResTypeToString(OpAndResType type) {
  if (type == HLO_AsyncBundle) return "HLO_AsyncBundle";
  if (type == HLO_ComplexTensor) return "HLO_ComplexTensor";
  if (type == HLO_DimensionTensor) return "HLO_DimensionTensor";
  if (type == HLO_Fp32Or64Tensor) return "HLO_Fp32Or64Tensor";
  if (type == HLO_FpOrComplexTensor) return "HLO_FpOrComplexTensor";
  if (type == HLO_FpTensor) return "HLO_FpTensor";
  if (type == HLO_IntFpOrComplexTensor) return "HLO_IntFpOrComplexTensor";
  if (type == HLO_IntOrFpTensor) return "HLO_IntOrFpTensor";
  if (type == HLO_IntTensor) return "HLO_IntTensor";
  if (type == HLO_PredIntOrFpTensor) return "HLO_PredIntOrFpTensor";
  if (type == HLO_PredOrIntTensor) return "HLO_PredOrIntTensor";
  if (type == HLO_PredTensor) return "HLO_PredTensor";
  if (type == HLO_QuantizedIntTensor) return "HLO_QuantizedIntTensor";
  if (type == HLO_ScalarIntTensor) return "HLO_ScalarIntTensor";
  if (type == HLO_StaticShapeTensor) return "HLO_StaticShapeTensor";
  if (type == HLO_Tensor) return "HLO_Tensor";
  if (type == HLO_TensorOrToken) return "HLO_TensorOrToken";
  if (type == HLO_TensorOrTokenOrTuple) return "HLO_TensorOrTokenOrTuple";
  if (type == HLO_Token) return "HLO_Token";
  if (type == HLO_Tuple) return "HLO_Tuple";
  if (type == I32Tensor) return "I32Tensor";
  if (type == Index) return "Index";
  if (type == Shape_WitnessType) return "Shape_WitnessType";
  if (type == anonymous_526) return "anonymous_526";
  if (type == anonymous_632) return "anonymous_632";
  if (type == anonymous_641) return "anonymous_641";
  if (type == anonymous_651) return "anonymous_651";
  if (type == anonymous_686) return "anonymous_686";
  if (type == anonymous_688) return "anonymous_688";
  if (type == anonymous_713) return "anonymous_713";
  if (type == anonymous_726) return "anonymous_726";
  if (type == anonymous_728) return "anonymous_728";
  if (type == anonymous_734) return "anonymous_734";
  if (type == anonymous_740) return "anonymous_740";
  if (type == anonymous_747) return "anonymous_747";
  if (type == anonymous_754) return "anonymous_754";
  assert(false && "Invalid OpAndResType");
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloArgResultAliasAttr() {
  std::vector<::llvm::SmallVector<int64_t>> argTupleIndicesEnumerants = genLlvmSmallVectorint64t();
  std::vector<int64_t> resultIndexEnumerants = genInt64t();
  std::vector<::llvm::SmallVector<int64_t>> resultTupleIndicesEnumerants = genLlvmSmallVectorint64t();
  std::vector<bool> isMustAliasEnumerants = genBool();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : argTupleIndicesEnumerants) {
    for (const auto &v1 : resultIndexEnumerants) {
      for (const auto &v2 : resultTupleIndicesEnumerants) {
        for (const auto &v3 : isMustAliasEnumerants) {
          ret.push_back(::mlir::mhlo::ArgResultAliasAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genChloComparisonDirectionAttr() {
  std::vector<::mlir::chlo::ComparisonDirection> valueEnumerants = {
    ::mlir::chlo::ComparisonDirection::EQ,
    ::mlir::chlo::ComparisonDirection::NE,
    ::mlir::chlo::ComparisonDirection::GE,
    ::mlir::chlo::ComparisonDirection::GT,
    ::mlir::chlo::ComparisonDirection::LE,
    ::mlir::chlo::ComparisonDirection::LT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::chlo::ComparisonDirectionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genChloComparisonTypeAttr() {
  std::vector<::mlir::chlo::ComparisonType> valueEnumerants = {
    ::mlir::chlo::ComparisonType::NOTYPE,
    ::mlir::chlo::ComparisonType::FLOAT,
    ::mlir::chlo::ComparisonType::TOTALORDER,
    ::mlir::chlo::ComparisonType::SIGNED,
    ::mlir::chlo::ComparisonType::UNSIGNED,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::chlo::ComparisonTypeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloChannelHandleAttr() {
  std::vector<int64_t> handleEnumerants = genInt64t();
  std::vector<int64_t> typeEnumerants = genInt64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : handleEnumerants) {
    for (const auto &v1 : typeEnumerants) {
      ret.push_back(::mlir::mhlo::ChannelHandleAttr::get(&ctx, 
        v0,
        v1));
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloConvDimensionNumbersAttr() {
  std::vector<int64_t> inputBatchDimensionEnumerants = genInt64t();
  std::vector<int64_t> inputFeatureDimensionEnumerants = genInt64t();
  std::vector<::llvm::SmallVector<int64_t>> inputSpatialDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<int64_t> kernelInputFeatureDimensionEnumerants = genInt64t();
  std::vector<int64_t> kernelOutputFeatureDimensionEnumerants = genInt64t();
  std::vector<::llvm::SmallVector<int64_t>> kernelSpatialDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<int64_t> outputBatchDimensionEnumerants = genInt64t();
  std::vector<int64_t> outputFeatureDimensionEnumerants = genInt64t();
  std::vector<::llvm::SmallVector<int64_t>> outputSpatialDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : inputBatchDimensionEnumerants) {
    for (const auto &v1 : inputFeatureDimensionEnumerants) {
      for (const auto &v2 : inputSpatialDimensionsEnumerants) {
        for (const auto &v3 : kernelInputFeatureDimensionEnumerants) {
          for (const auto &v4 : kernelOutputFeatureDimensionEnumerants) {
            for (const auto &v5 : kernelSpatialDimensionsEnumerants) {
              for (const auto &v6 : outputBatchDimensionEnumerants) {
                for (const auto &v7 : outputFeatureDimensionEnumerants) {
                  for (const auto &v8 : outputSpatialDimensionsEnumerants) {
                    ret.push_back(::mlir::mhlo::ConvDimensionNumbersAttr::get(&ctx, 
                      v0,
                      v1,
                      v2,
                      v3,
                      v4,
                      v5,
                      v6,
                      v7,
                      v8));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloDotDimensionNumbersAttr() {
  std::vector<::llvm::SmallVector<int64_t>> lhsBatchingDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> rhsBatchingDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> lhsContractingDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> rhsContractingDimensionsEnumerants = genLlvmSmallVectorint64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : lhsBatchingDimensionsEnumerants) {
    for (const auto &v1 : rhsBatchingDimensionsEnumerants) {
      for (const auto &v2 : lhsContractingDimensionsEnumerants) {
        for (const auto &v3 : rhsContractingDimensionsEnumerants) {
          ret.push_back(::mlir::mhlo::DotDimensionNumbersAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloGatherDimensionNumbersAttr() {
  std::vector<::llvm::SmallVector<int64_t>> offsetDimsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> collapsedSliceDimsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> startIndexMapEnumerants = genLlvmSmallVectorint64t();
  std::vector<int64_t> indexVectorDimEnumerants = genInt64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : offsetDimsEnumerants) {
    for (const auto &v1 : collapsedSliceDimsEnumerants) {
      for (const auto &v2 : startIndexMapEnumerants) {
        for (const auto &v3 : indexVectorDimEnumerants) {
          ret.push_back(::mlir::mhlo::GatherDimensionNumbersAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloComparisonDirectionAttr() {
  std::vector<::mlir::mhlo::ComparisonDirection> valueEnumerants = {
    ::mlir::mhlo::ComparisonDirection::EQ,
    ::mlir::mhlo::ComparisonDirection::NE,
    ::mlir::mhlo::ComparisonDirection::GE,
    ::mlir::mhlo::ComparisonDirection::GT,
    ::mlir::mhlo::ComparisonDirection::LE,
    ::mlir::mhlo::ComparisonDirection::LT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::ComparisonDirectionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloComparisonTypeAttr() {
  std::vector<::mlir::mhlo::ComparisonType> valueEnumerants = {
    ::mlir::mhlo::ComparisonType::NOTYPE,
    ::mlir::mhlo::ComparisonType::FLOAT,
    ::mlir::mhlo::ComparisonType::TOTALORDER,
    ::mlir::mhlo::ComparisonType::SIGNED,
    ::mlir::mhlo::ComparisonType::UNSIGNED,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::ComparisonTypeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloDequantizeModeAttr() {
  std::vector<::mlir::mhlo::DequantizeMode> valueEnumerants = {
    ::mlir::mhlo::DequantizeMode::MIN_COMBINED,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::DequantizeModeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloDomainKindAttr() {
  std::vector<::mlir::mhlo::DomainKind> valueEnumerants = {
    ::mlir::mhlo::DomainKind::sharding,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::DomainKindAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloFftTypeAttr() {
  std::vector<::mlir::mhlo::FftType> valueEnumerants = {
    ::mlir::mhlo::FftType::FFT,
    ::mlir::mhlo::FftType::IFFT,
    ::mlir::mhlo::FftType::RFFT,
    ::mlir::mhlo::FftType::IRFFT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::FftTypeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloFusionKindAttr() {
  std::vector<::mlir::mhlo::FusionKind> valueEnumerants = {
    ::mlir::mhlo::FusionKind::kLoop,
    ::mlir::mhlo::FusionKind::kInput,
    ::mlir::mhlo::FusionKind::kOutput,
    ::mlir::mhlo::FusionKind::kCustom,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::FusionKindAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloPrecisionAttr() {
  std::vector<::mlir::mhlo::Precision> valueEnumerants = {
    ::mlir::mhlo::Precision::DEFAULT,
    ::mlir::mhlo::Precision::HIGH,
    ::mlir::mhlo::Precision::HIGHEST,
    ::mlir::mhlo::Precision::PACKED_NIBBLE,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::PrecisionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloRngAlgorithmAttr() {
  std::vector<::mlir::mhlo::RngAlgorithm> valueEnumerants = {
    ::mlir::mhlo::RngAlgorithm::DEFAULT,
    ::mlir::mhlo::RngAlgorithm::THREE_FRY,
    ::mlir::mhlo::RngAlgorithm::PHILOX,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::RngAlgorithmAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloRngDistributionAttr() {
  std::vector<::mlir::mhlo::RngDistribution> valueEnumerants = {
    ::mlir::mhlo::RngDistribution::UNIFORM,
    ::mlir::mhlo::RngDistribution::NORMAL,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::RngDistributionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloTransposeAttr() {
  std::vector<::mlir::mhlo::Transpose> valueEnumerants = {
    ::mlir::mhlo::Transpose::TRANSPOSE_INVALID,
    ::mlir::mhlo::Transpose::NO_TRANSPOSE,
    ::mlir::mhlo::Transpose::TRANSPOSE,
    ::mlir::mhlo::Transpose::ADJOINT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::mhlo::TransposeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloOutputOperandAliasAttr() {
  std::vector<::llvm::SmallVector<int64_t>> outputTupleIndicesEnumerants = genLlvmSmallVectorint64t();
  std::vector<int64_t> operandIndexEnumerants = genInt64t();
  std::vector<::llvm::SmallVector<int64_t>> operandTupleIndicesEnumerants = genLlvmSmallVectorint64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : outputTupleIndicesEnumerants) {
    for (const auto &v1 : operandIndexEnumerants) {
      for (const auto &v2 : operandTupleIndicesEnumerants) {
        ret.push_back(::mlir::mhlo::OutputOperandAliasAttr::get(&ctx, 
          v0,
          v1,
          v2));
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloScatterDimensionNumbersAttr() {
  std::vector<::llvm::SmallVector<int64_t>> updateWindowDimsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> insertedWindowDimsEnumerants = genLlvmSmallVectorint64t();
  std::vector<::llvm::SmallVector<int64_t>> scatterDimsToOperandDimsEnumerants = genLlvmSmallVectorint64t();
  std::vector<int64_t> indexVectorDimEnumerants = genInt64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : updateWindowDimsEnumerants) {
    for (const auto &v1 : insertedWindowDimsEnumerants) {
      for (const auto &v2 : scatterDimsToOperandDimsEnumerants) {
        for (const auto &v3 : indexVectorDimEnumerants) {
          ret.push_back(::mlir::mhlo::ScatterDimensionNumbersAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genMhloTypeExtensionsAttr() {
  std::vector<::llvm::SmallVector<int64_t>> boundsEnumerants = genLlvmSmallVectorint64t();
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : boundsEnumerants) {
    ret.push_back(::mlir::mhlo::TypeExtensionsAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genArrayAttr() {
  llvm::outs() << "WARNING: Not implemented: genArrayAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genBoolAttr() {
  llvm::outs() << "WARNING: Not implemented: genBoolAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genChannelHandleAttr() {
  llvm::outs() << "WARNING: Not implemented: genChannelHandleAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genComparisonTypeAttr() {
  llvm::outs() << "WARNING: Not implemented: genComparisonTypeAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genCustomCallApiVersionAttr() {
  llvm::outs() << "WARNING: Not implemented: genCustomCallApiVersionAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genDenseElementsAttr() {
  llvm::outs() << "WARNING: Not implemented: genDenseElementsAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genDenseIntElementsAttr() {
  llvm::outs() << "WARNING: Not implemented: genDenseIntElementsAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genElementsAttr() {
  llvm::outs() << "WARNING: Not implemented: genElementsAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genFlatSymbolRefAttr() {
  llvm::outs() << "WARNING: Not implemented: genFlatSymbolRefAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genFloatAttr() {
  llvm::outs() << "WARNING: Not implemented: genFloatAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genFusionKindAttr() {
  llvm::outs() << "WARNING: Not implemented: genFusionKindAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genIntegerAttr() {
  llvm::outs() << "WARNING: Not implemented: genIntegerAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genStringAttr() {
  llvm::outs() << "WARNING: Not implemented: genStringAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genTypedAttr() {
  llvm::outs() << "WARNING: Not implemented: genTypedAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGenerator::genUnitAttr() {
  llvm::outs() << "WARNING: Not implemented: genUnitAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<::llvm::SmallVector<int64_t>> AttributeGenerator::genLlvmSmallVectorint64t() {
  llvm::outs() << "WARNING: Not implemented: genLlvmSmallVectorint64t\n";
  std::vector<::llvm::SmallVector<int64_t>> ret;
  return ret;
}

std::vector<bool> AttributeGenerator::genBool() {
  llvm::outs() << "WARNING: Not implemented: genBool\n";
  std::vector<bool> ret;
  return ret;
}

std::vector<int64_t> AttributeGenerator::genInt64t() {
  llvm::outs() << "WARNING: Not implemented: genInt64t\n";
  std::vector<int64_t> ret;
  return ret;
}

class chlo_acos : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_acosh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_asin : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_asinh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_atan : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_atanh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_bessel_i1e : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_add : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_and : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_atan2 : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_compare : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::chlo::ComparisonDirectionAttr();
      case 2: return ::mlir::chlo::ComparisonTypeAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
      case 1: return "comparison_direction";
      case 2: return "compare_type";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genChloComparisonDirectionAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genComparisonTypeAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_complex : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_divide : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_maximum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_minimum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_multiply : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_next_after : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_or : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_polygamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_power : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_remainder : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_select : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_left : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_right_arithmetic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_right_logical : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_subtract : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_xor : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_zeta : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_conj : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_constant_like : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::TypedAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "value";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genTypedAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_constant : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "value";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_cosh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_digamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_dynamic_reshape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_erf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_erfc : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_inf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_neg_inf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_pos_inf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_lgamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_minimum_broadcast_shapes : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_526;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_526;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_next_after : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_polygamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_rank_specialization_cluster : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_rank_specialization_cluster_yield : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class chlo_sinh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_tan : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_top_k : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 2; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "k";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_zeta : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_abs : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_632;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_632;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_add_dependency : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
      case 1: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_add : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_after_all : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_all_gather : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::mhlo::ChannelHandleAttr();
      case 3: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "all_gather_dim";
      case 1: return "replica_groups";
      case 2: return "channel_handle";
      case 3: return "use_global_device_ids";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genChannelHandleAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    // auto attr3 = attrGen->genUnitAttr();
    // attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_all_reduce : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::mhlo::ChannelHandleAttr();
      case 2: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "replica_groups";
      case 1: return "channel_handle";
      case 2: return "use_global_device_ids";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genChannelHandleAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genUnitAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_all_to_all : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::IntegerAttr();
      case 2: return ::mlir::IntegerAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "split_dimension";
      case 1: return "concat_dimension";
      case 2: return "split_count";
      case 3: return "replica_groups";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    auto attr2 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    auto attr3 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_and : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_async_done : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_AsyncBundle;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FlatSymbolRefAttr();
      case 1: return ::mlir::StringAttr();
      case 2: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "called_computation";
      case 1: return "execution_thread";
      case 2: return "group_id";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genFlatSymbolRefAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genIntegerAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_async_start : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FlatSymbolRefAttr();
      case 1: return ::mlir::StringAttr();
      case 2: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "called_computation";
      case 1: return "execution_thread";
      case 2: return "group_id";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genFlatSymbolRefAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genIntegerAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_AsyncBundle;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_async_update : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_AsyncBundle;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FlatSymbolRefAttr();
      case 1: return ::mlir::StringAttr();
      case 2: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "called_computation";
      case 1: return "execution_thread";
      case 2: return "group_id";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genFlatSymbolRefAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genIntegerAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_AsyncBundle;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_atan2 : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_batch_norm_grad : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 5; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 3; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
      case 1: return anonymous_688;
      case 2: return anonymous_688;
      case 3: return anonymous_688;
      case 4: return anonymous_686;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FloatAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "epsilon";
      case 1: return "feature_index";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genFloatAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
      case 1: return anonymous_688;
      case 2: return anonymous_688;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_batch_norm_inference : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 5; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
      case 1: return anonymous_688;
      case 2: return anonymous_688;
      case 3: return anonymous_688;
      case 4: return anonymous_688;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FloatAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "epsilon";
      case 1: return "feature_index";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genFloatAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_batch_norm_training : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 3; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
      case 1: return anonymous_688;
      case 2: return anonymous_688;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FloatAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "epsilon";
      case 1: return "feature_index";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genFloatAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
      case 1: return anonymous_688;
      case 2: return anonymous_688;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_bitcast_convert : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_bitcast : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_broadcast_in_dim : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_broadcast : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_sizes";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_case : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return I32Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cbrt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_ceil : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cholesky : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "lower";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_clamp : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_count_leading_zeros : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_collective_permute : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::mhlo::ChannelHandleAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "source_target_pairs";
      case 1: return "channel_handle";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genChannelHandleAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_compare : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::ComparisonDirectionAttr();
      case 1: return ::mlir::mhlo::ComparisonTypeAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "comparison_direction";
      case 1: return "compare_type";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloComparisonDirectionAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genComparisonTypeAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_complex : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Fp32Or64Tensor;
      case 1: return HLO_Fp32Or64Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_compute_reshape_shape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return Index;
      case 1: return anonymous_754;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_754;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_concatenate : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_constant : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "value";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_convert : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_convolution : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 9; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
      case 4: return ::mlir::DenseElementsAttr();
      case 5: return ::mlir::mhlo::ConvDimensionNumbersAttr();
      case 6: return ::mlir::IntegerAttr();
      case 7: return ::mlir::IntegerAttr();
      case 8: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_strides";
      case 1: return "padding";
      case 2: return "lhs_dilation";
      case 3: return "rhs_dilation";
      case 4: return "window_reversal";
      case 5: return "dimension_numbers";
      case 6: return "feature_group_count";
      case 7: return "batch_group_count";
      case 8: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    // auto attr3 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    // auto attr4 = attrGen->genDenseElementsAttr();
    // attrs.insert(attrs.end(), attr4.begin(), attr4.end());
    auto attr5 = attrGen->genMhloConvDimensionNumbersAttr();
    attrs.insert(attrs.end(), attr5.begin(), attr5.end());
    auto attr6 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr6.begin(), attr6.end());
    auto attr7 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr7.begin(), attr7.end());
    // auto attr8 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr8.begin(), attr8.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_copy : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "is_cross_program_prefetch";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genUnitAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cosine : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_create_token : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cross_replica_sum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "replica_groups";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cstr_reshapable : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return Index;
      case 1: return anonymous_754;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return Shape_WitnessType;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_custom_call : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 8; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
      case 1: return ::mlir::BoolAttr();
      case 2: return ::mlir::StringAttr();
      case 3: return ::mlir::mhlo::CustomCallApiVersionAttr();
      case 4: return ::mlir::ArrayAttr();
      case 5: return ::mlir::ArrayAttr();
      case 6: return ::mlir::ArrayAttr();
      case 7: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "call_target_name";
      case 1: return "has_side_effect";
      case 2: return "backend_config";
      case 3: return "api_version";
      case 4: return "called_computations";
      case 5: return "operand_layouts";
      case 6: return "result_layouts";
      case 7: return "output_operand_aliases";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genStringAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    // auto attr3 = attrGen->genCustomCallApiVersionAttr();
    // attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    // auto attr4 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr4.begin(), attr4.end());
    // auto attr5 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr5.begin(), attr5.end());
    // auto attr6 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr6.begin(), attr6.end());
    // auto attr7 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr7.begin(), attr7.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_divide : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_domain : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::DomainKindAttr();
      case 1: return ::mlir::StringAttr();
      case 2: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "kind";
      case 1: return "entry_metadata";
      case 2: return "exit_metadata";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloDomainKindAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    auto attr2 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dot_general : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::DotDimensionNumbersAttr();
      case 1: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dot_dimension_numbers";
      case 1: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloDotDimensionNumbersAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dot : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_broadcast_in_dim : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
      case 1: return "known_expanding_dimensions";
      case 2: return "known_nonexpanding_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_conv : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 9; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
      case 4: return ::mlir::DenseElementsAttr();
      case 5: return ::mlir::mhlo::ConvDimensionNumbersAttr();
      case 6: return ::mlir::IntegerAttr();
      case 7: return ::mlir::IntegerAttr();
      case 8: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_strides";
      case 1: return "padding";
      case 2: return "lhs_dilation";
      case 3: return "rhs_dilation";
      case 4: return "window_reversal";
      case 5: return "dimension_numbers";
      case 6: return "feature_group_count";
      case 7: return "batch_group_count";
      case 8: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    // auto attr3 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    // auto attr4 = attrGen->genDenseElementsAttr();
    // attrs.insert(attrs.end(), attr4.begin(), attr4.end());
    auto attr5 = attrGen->genMhloConvDimensionNumbersAttr();
    attrs.insert(attrs.end(), attr5.begin(), attr5.end());
    auto attr6 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr6.begin(), attr6.end());
    auto attr7 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr7.begin(), attr7.end());
    // auto attr8 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr8.begin(), attr8.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_gather : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_IntTensor;
      case 2: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::GatherDimensionNumbersAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension_numbers";
      case 1: return "indices_are_sorted";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloGatherDimensionNumbersAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_iota : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "iota_dimension";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_pad : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 5; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_DimensionTensor;
      case 3: return HLO_DimensionTensor;
      case 4: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_reshape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_ScalarIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "slice_sizes";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_update_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_ScalarIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_einsum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "einsum_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_exponential : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_exponential_minus_one : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_fft : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::FftTypeAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "fft_type";
      case 1: return "fft_length";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloFftTypeAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_floor : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_fusion : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::FusionKindAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "fusion_kind";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genFusionKindAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_747;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_gather : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::GatherDimensionNumbersAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension_numbers";
      case 1: return "slice_sizes";
      case 2: return "indices_are_sorted";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloGatherDimensionNumbersAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_get_dimension_size : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return I32Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_get_tuple_element : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tuple;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "index";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_if : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 2; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_imag : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_infeed : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
      case 1: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "infeed_config";
      case 1: return "layout";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genArrayAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_iota : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "iota_dimension";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_is_finite : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_log_plus_one : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_log : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_logistic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_map : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_maximum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_minimum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_multiply : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_negate : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_not : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_optimization_barrier : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_or : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_outfeed : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "outfeed_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_pad : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "edge_padding_low";
      case 1: return "edge_padding_high";
      case 2: return "interior_padding";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    auto attr2 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_partition_id : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_651;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_popcnt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_power : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_real_dynamic_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 4; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
      case 2: return HLO_DimensionTensor;
      case 3: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_real : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_recv : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::ChannelHandleAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "channel_handle";
      case 1: return "is_host_transfer";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloChannelHandleAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce_precision : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "exponent_bits";
      case 1: return "mantissa_bits";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce_scatter : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::mhlo::ChannelHandleAttr();
      case 3: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "scatter_dimension";
      case 1: return "replica_groups";
      case 2: return "channel_handle";
      case 3: return "use_global_device_ids";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genChannelHandleAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    // auto attr3 = attrGen->genUnitAttr();
    // attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce_window : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 5; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
      case 4: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_dimensions";
      case 1: return "window_strides";
      case 2: return "base_dilations";
      case 3: return "window_dilations";
      case 4: return "padding";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    // auto attr3 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    // auto attr4 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr4.begin(), attr4.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_remainder : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_replica_id : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_651;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reshape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_return : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reverse : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_rng_bit_generator : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 2; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntOrFpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::RngAlgorithmAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "rng_algorithm";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloRngAlgorithmAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntOrFpTensor;
      case 1: return HLO_IntOrFpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_rng : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_726;
      case 1: return anonymous_726;
      case 2: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::RngDistributionAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "rng_distribution";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloRngDistributionAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredIntOrFpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_round_nearest_even : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_round_nearest_afz : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_rsqrt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_scatter : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return anonymous_713;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::ScatterDimensionNumbersAttr();
      case 1: return ::mlir::BoolAttr();
      case 2: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "scatter_dimension_numbers";
      case 1: return "indices_are_sorted";
      case 2: return "unique_indices";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloScatterDimensionNumbersAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_select_and_scatter : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 2; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_dimensions";
      case 1: return "window_strides";
      case 2: return "padding";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    // auto attr2 = attrGen->genDenseIntElementsAttr();
    // attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_select : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_send : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::mhlo::ChannelHandleAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "channel_handle";
      case 1: return "is_host_transfer";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genMhloChannelHandleAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_set_dimension_size : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return I32Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_shift_left : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_shift_right_arithmetic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_shift_right_logical : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sign : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_632;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_632;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sine : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "start_indices";
      case 1: return "limit_indices";
      case 2: return "strides";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    auto attr2 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sort : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
      case 1: return "is_stable";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    // auto attr0 = attrGen->genIntegerAttr();
    // attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    // auto attr1 = attrGen->genBoolAttr();
    // attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sqrt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_stochastic_convert : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return anonymous_641;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_subtract : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_tanh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_torch_index_select : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dim";
      case 1: return "batch_dims";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_trace : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "tag";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_transpose : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "permutation";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genDenseIntElementsAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_triangular_solve : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
      case 1: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::BoolAttr();
      case 1: return ::mlir::BoolAttr();
      case 2: return ::mlir::BoolAttr();
      case 3: return ::mlir::mhlo::TransposeAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "left_side";
      case 1: return "lower";
      case 2: return "unit_diagonal";
      case 3: return "transpose_a";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genBoolAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    auto attr1 = attrGen->genBoolAttr();
    attrs.insert(attrs.end(), attr1.begin(), attr1.end());
    auto attr2 = attrGen->genBoolAttr();
    attrs.insert(attrs.end(), attr2.begin(), attr2.end());
    auto attr3 = attrGen->genMhloTransposeAttr();
    attrs.insert(attrs.end(), attr3.begin(), attr3.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_tuple : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_unary_einsum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "einsum_config";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genStringAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_uniform_dequantize : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_QuantizedIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_740;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_uniform_quantize : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_734;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_QuantizedIntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_while : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 2; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_xla_rng_get_and_update_state : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "delta";
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    auto attr0 = attrGen->genIntegerAttr();
    attrs.insert(attrs.end(), attr0.begin(), attr0.end());
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_728;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_xor : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<mlir::Attribute> genAttributes(AttributeGeneratorPtr &attrGen) const override {
    std::vector<mlir::Attribute> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

GrammarOpPtr createGrammarOp(std::string name) {
  if (name == "chlo.acos")
    return std::make_unique<chlo_acos>();
  if (name == "chlo.acosh")
    return std::make_unique<chlo_acosh>();
  if (name == "chlo.asin")
    return std::make_unique<chlo_asin>();
  if (name == "chlo.asinh")
    return std::make_unique<chlo_asinh>();
  if (name == "chlo.atan")
    return std::make_unique<chlo_atan>();
  if (name == "chlo.atanh")
    return std::make_unique<chlo_atanh>();
  if (name == "chlo.bessel_i1e")
    return std::make_unique<chlo_bessel_i1e>();
  if (name == "chlo.broadcast_add")
    return std::make_unique<chlo_broadcast_add>();
  if (name == "chlo.broadcast_and")
    return std::make_unique<chlo_broadcast_and>();
  if (name == "chlo.broadcast_atan2")
    return std::make_unique<chlo_broadcast_atan2>();
  if (name == "chlo.broadcast_compare")
    return std::make_unique<chlo_broadcast_compare>();
  if (name == "chlo.broadcast_complex")
    return std::make_unique<chlo_broadcast_complex>();
  if (name == "chlo.broadcast_divide")
    return std::make_unique<chlo_broadcast_divide>();
  if (name == "chlo.broadcast_maximum")
    return std::make_unique<chlo_broadcast_maximum>();
  if (name == "chlo.broadcast_minimum")
    return std::make_unique<chlo_broadcast_minimum>();
  if (name == "chlo.broadcast_multiply")
    return std::make_unique<chlo_broadcast_multiply>();
  if (name == "chlo.broadcast_next_after")
    return std::make_unique<chlo_broadcast_next_after>();
  if (name == "chlo.broadcast_or")
    return std::make_unique<chlo_broadcast_or>();
  if (name == "chlo.broadcast_polygamma")
    return std::make_unique<chlo_broadcast_polygamma>();
  if (name == "chlo.broadcast_power")
    return std::make_unique<chlo_broadcast_power>();
  if (name == "chlo.broadcast_remainder")
    return std::make_unique<chlo_broadcast_remainder>();
  if (name == "chlo.broadcast_select")
    return std::make_unique<chlo_broadcast_select>();
  if (name == "chlo.broadcast_shift_left")
    return std::make_unique<chlo_broadcast_shift_left>();
  if (name == "chlo.broadcast_shift_right_arithmetic")
    return std::make_unique<chlo_broadcast_shift_right_arithmetic>();
  if (name == "chlo.broadcast_shift_right_logical")
    return std::make_unique<chlo_broadcast_shift_right_logical>();
  if (name == "chlo.broadcast_subtract")
    return std::make_unique<chlo_broadcast_subtract>();
  if (name == "chlo.broadcast_xor")
    return std::make_unique<chlo_broadcast_xor>();
  if (name == "chlo.broadcast_zeta")
    return std::make_unique<chlo_broadcast_zeta>();
  if (name == "chlo.conj")
    return std::make_unique<chlo_conj>();
  if (name == "chlo.constant_like")
    return std::make_unique<chlo_constant_like>();
  if (name == "chlo.constant")
    return std::make_unique<chlo_constant>();
  if (name == "chlo.cosh")
    return std::make_unique<chlo_cosh>();
  if (name == "chlo.digamma")
    return std::make_unique<chlo_digamma>();
  if (name == "chlo.dynamic_reshape")
    return std::make_unique<chlo_dynamic_reshape>();
  if (name == "chlo.erf")
    return std::make_unique<chlo_erf>();
  if (name == "chlo.erfc")
    return std::make_unique<chlo_erfc>();
  if (name == "chlo.is_inf")
    return std::make_unique<chlo_is_inf>();
  if (name == "chlo.is_neg_inf")
    return std::make_unique<chlo_is_neg_inf>();
  if (name == "chlo.is_pos_inf")
    return std::make_unique<chlo_is_pos_inf>();
  if (name == "chlo.lgamma")
    return std::make_unique<chlo_lgamma>();
  if (name == "chlo.minimum_broadcast_shapes")
    return std::make_unique<chlo_minimum_broadcast_shapes>();
  if (name == "chlo.next_after")
    return std::make_unique<chlo_next_after>();
  if (name == "chlo.polygamma")
    return std::make_unique<chlo_polygamma>();
  if (name == "chlo.rank_specialization_cluster")
    return std::make_unique<chlo_rank_specialization_cluster>();
  if (name == "chlo.rank_specialization_cluster_yield")
    return std::make_unique<chlo_rank_specialization_cluster_yield>();
  if (name == "chlo.sinh")
    return std::make_unique<chlo_sinh>();
  if (name == "chlo.tan")
    return std::make_unique<chlo_tan>();
  if (name == "chlo.top_k")
    return std::make_unique<chlo_top_k>();
  if (name == "chlo.zeta")
    return std::make_unique<chlo_zeta>();
  if (name == "mhlo.abs")
    return std::make_unique<mhlo_abs>();
  if (name == "mhlo.add_dependency")
    return std::make_unique<mhlo_add_dependency>();
  if (name == "mhlo.add")
    return std::make_unique<mhlo_add>();
  if (name == "mhlo.after_all")
    return std::make_unique<mhlo_after_all>();
  if (name == "mhlo.all_gather")
    return std::make_unique<mhlo_all_gather>();
  if (name == "mhlo.all_reduce")
    return std::make_unique<mhlo_all_reduce>();
  if (name == "mhlo.all_to_all")
    return std::make_unique<mhlo_all_to_all>();
  if (name == "mhlo.and")
    return std::make_unique<mhlo_and>();
  if (name == "mhlo.async_done")
    return std::make_unique<mhlo_async_done>();
  if (name == "mhlo.async_start")
    return std::make_unique<mhlo_async_start>();
  if (name == "mhlo.async_update")
    return std::make_unique<mhlo_async_update>();
  if (name == "mhlo.atan2")
    return std::make_unique<mhlo_atan2>();
  if (name == "mhlo.batch_norm_grad")
    return std::make_unique<mhlo_batch_norm_grad>();
  if (name == "mhlo.batch_norm_inference")
    return std::make_unique<mhlo_batch_norm_inference>();
  if (name == "mhlo.batch_norm_training")
    return std::make_unique<mhlo_batch_norm_training>();
  if (name == "mhlo.bitcast_convert")
    return std::make_unique<mhlo_bitcast_convert>();
  if (name == "mhlo.bitcast")
    return std::make_unique<mhlo_bitcast>();
  if (name == "mhlo.broadcast_in_dim")
    return std::make_unique<mhlo_broadcast_in_dim>();
  if (name == "mhlo.broadcast")
    return std::make_unique<mhlo_broadcast>();
  if (name == "mhlo.case")
    return std::make_unique<mhlo_case>();
  if (name == "mhlo.cbrt")
    return std::make_unique<mhlo_cbrt>();
  if (name == "mhlo.ceil")
    return std::make_unique<mhlo_ceil>();
  if (name == "mhlo.cholesky")
    return std::make_unique<mhlo_cholesky>();
  if (name == "mhlo.clamp")
    return std::make_unique<mhlo_clamp>();
  if (name == "mhlo.count_leading_zeros")
    return std::make_unique<mhlo_count_leading_zeros>();
  if (name == "mhlo.collective_permute")
    return std::make_unique<mhlo_collective_permute>();
  if (name == "mhlo.compare")
    return std::make_unique<mhlo_compare>();
  if (name == "mhlo.complex")
    return std::make_unique<mhlo_complex>();
  if (name == "mhlo.compute_reshape_shape")
    return std::make_unique<mhlo_compute_reshape_shape>();
  if (name == "mhlo.concatenate")
    return std::make_unique<mhlo_concatenate>();
  if (name == "mhlo.constant")
    return std::make_unique<mhlo_constant>();
  if (name == "mhlo.convert")
    return std::make_unique<mhlo_convert>();
  if (name == "mhlo.convolution")
    return std::make_unique<mhlo_convolution>();
  if (name == "mhlo.copy")
    return std::make_unique<mhlo_copy>();
  if (name == "mhlo.cosine")
    return std::make_unique<mhlo_cosine>();
  if (name == "mhlo.create_token")
    return std::make_unique<mhlo_create_token>();
  if (name == "mhlo.cross-replica-sum")
    return std::make_unique<mhlo_cross_replica_sum>();
  if (name == "mhlo.cstr_reshapable")
    return std::make_unique<mhlo_cstr_reshapable>();
  if (name == "mhlo.custom_call")
    return std::make_unique<mhlo_custom_call>();
  if (name == "mhlo.divide")
    return std::make_unique<mhlo_divide>();
  if (name == "mhlo.domain")
    return std::make_unique<mhlo_domain>();
  if (name == "mhlo.dot_general")
    return std::make_unique<mhlo_dot_general>();
  if (name == "mhlo.dot")
    return std::make_unique<mhlo_dot>();
  if (name == "mhlo.dynamic_broadcast_in_dim")
    return std::make_unique<mhlo_dynamic_broadcast_in_dim>();
  if (name == "mhlo.dynamic_conv")
    return std::make_unique<mhlo_dynamic_conv>();
  if (name == "mhlo.dynamic_gather")
    return std::make_unique<mhlo_dynamic_gather>();
  if (name == "mhlo.dynamic_iota")
    return std::make_unique<mhlo_dynamic_iota>();
  if (name == "mhlo.dynamic_pad")
    return std::make_unique<mhlo_dynamic_pad>();
  if (name == "mhlo.dynamic_reshape")
    return std::make_unique<mhlo_dynamic_reshape>();
  if (name == "mhlo.dynamic_slice")
    return std::make_unique<mhlo_dynamic_slice>();
  if (name == "mhlo.dynamic_update_slice")
    return std::make_unique<mhlo_dynamic_update_slice>();
  if (name == "mhlo.einsum")
    return std::make_unique<mhlo_einsum>();
  if (name == "mhlo.exponential")
    return std::make_unique<mhlo_exponential>();
  if (name == "mhlo.exponential_minus_one")
    return std::make_unique<mhlo_exponential_minus_one>();
  if (name == "mhlo.fft")
    return std::make_unique<mhlo_fft>();
  if (name == "mhlo.floor")
    return std::make_unique<mhlo_floor>();
  if (name == "mhlo.fusion")
    return std::make_unique<mhlo_fusion>();
  if (name == "mhlo.gather")
    return std::make_unique<mhlo_gather>();
  if (name == "mhlo.get_dimension_size")
    return std::make_unique<mhlo_get_dimension_size>();
  if (name == "mhlo.get_tuple_element")
    return std::make_unique<mhlo_get_tuple_element>();
  if (name == "mhlo.if")
    return std::make_unique<mhlo_if>();
  if (name == "mhlo.imag")
    return std::make_unique<mhlo_imag>();
  if (name == "mhlo.infeed")
    return std::make_unique<mhlo_infeed>();
  if (name == "mhlo.iota")
    return std::make_unique<mhlo_iota>();
  if (name == "mhlo.is_finite")
    return std::make_unique<mhlo_is_finite>();
  if (name == "mhlo.log_plus_one")
    return std::make_unique<mhlo_log_plus_one>();
  if (name == "mhlo.log")
    return std::make_unique<mhlo_log>();
  if (name == "mhlo.logistic")
    return std::make_unique<mhlo_logistic>();
  if (name == "mhlo.map")
    return std::make_unique<mhlo_map>();
  if (name == "mhlo.maximum")
    return std::make_unique<mhlo_maximum>();
  if (name == "mhlo.minimum")
    return std::make_unique<mhlo_minimum>();
  if (name == "mhlo.multiply")
    return std::make_unique<mhlo_multiply>();
  if (name == "mhlo.negate")
    return std::make_unique<mhlo_negate>();
  if (name == "mhlo.not")
    return std::make_unique<mhlo_not>();
  if (name == "mhlo.optimization_barrier")
    return std::make_unique<mhlo_optimization_barrier>();
  if (name == "mhlo.or")
    return std::make_unique<mhlo_or>();
  if (name == "mhlo.outfeed")
    return std::make_unique<mhlo_outfeed>();
  if (name == "mhlo.pad")
    return std::make_unique<mhlo_pad>();
  if (name == "mhlo.partition_id")
    return std::make_unique<mhlo_partition_id>();
  if (name == "mhlo.popcnt")
    return std::make_unique<mhlo_popcnt>();
  if (name == "mhlo.power")
    return std::make_unique<mhlo_power>();
  if (name == "mhlo.real_dynamic_slice")
    return std::make_unique<mhlo_real_dynamic_slice>();
  if (name == "mhlo.real")
    return std::make_unique<mhlo_real>();
  if (name == "mhlo.recv")
    return std::make_unique<mhlo_recv>();
  if (name == "mhlo.reduce")
    return std::make_unique<mhlo_reduce>();
  if (name == "mhlo.reduce_precision")
    return std::make_unique<mhlo_reduce_precision>();
  if (name == "mhlo.reduce_scatter")
    return std::make_unique<mhlo_reduce_scatter>();
  if (name == "mhlo.reduce_window")
    return std::make_unique<mhlo_reduce_window>();
  if (name == "mhlo.remainder")
    return std::make_unique<mhlo_remainder>();
  if (name == "mhlo.replica_id")
    return std::make_unique<mhlo_replica_id>();
  if (name == "mhlo.reshape")
    return std::make_unique<mhlo_reshape>();
  if (name == "mhlo.return")
    return std::make_unique<mhlo_return>();
  if (name == "mhlo.reverse")
    return std::make_unique<mhlo_reverse>();
  if (name == "mhlo.rng_bit_generator")
    return std::make_unique<mhlo_rng_bit_generator>();
  if (name == "mhlo.rng")
    return std::make_unique<mhlo_rng>();
  if (name == "mhlo.round_nearest_even")
    return std::make_unique<mhlo_round_nearest_even>();
  if (name == "mhlo.round_nearest_afz")
    return std::make_unique<mhlo_round_nearest_afz>();
  if (name == "mhlo.rsqrt")
    return std::make_unique<mhlo_rsqrt>();
  if (name == "mhlo.scatter")
    return std::make_unique<mhlo_scatter>();
  if (name == "mhlo.select_and_scatter")
    return std::make_unique<mhlo_select_and_scatter>();
  if (name == "mhlo.select")
    return std::make_unique<mhlo_select>();
  if (name == "mhlo.send")
    return std::make_unique<mhlo_send>();
  if (name == "mhlo.set_dimension_size")
    return std::make_unique<mhlo_set_dimension_size>();
  if (name == "mhlo.shift_left")
    return std::make_unique<mhlo_shift_left>();
  if (name == "mhlo.shift_right_arithmetic")
    return std::make_unique<mhlo_shift_right_arithmetic>();
  if (name == "mhlo.shift_right_logical")
    return std::make_unique<mhlo_shift_right_logical>();
  if (name == "mhlo.sign")
    return std::make_unique<mhlo_sign>();
  if (name == "mhlo.sine")
    return std::make_unique<mhlo_sine>();
  if (name == "mhlo.slice")
    return std::make_unique<mhlo_slice>();
  if (name == "mhlo.sort")
    return std::make_unique<mhlo_sort>();
  if (name == "mhlo.sqrt")
    return std::make_unique<mhlo_sqrt>();
  if (name == "mhlo.stochastic_convert")
    return std::make_unique<mhlo_stochastic_convert>();
  if (name == "mhlo.subtract")
    return std::make_unique<mhlo_subtract>();
  if (name == "mhlo.tanh")
    return std::make_unique<mhlo_tanh>();
  if (name == "mhlo.torch_index_select")
    return std::make_unique<mhlo_torch_index_select>();
  if (name == "mhlo.trace")
    return std::make_unique<mhlo_trace>();
  if (name == "mhlo.transpose")
    return std::make_unique<mhlo_transpose>();
  if (name == "mhlo.triangular_solve")
    return std::make_unique<mhlo_triangular_solve>();
  if (name == "mhlo.tuple")
    return std::make_unique<mhlo_tuple>();
  if (name == "mhlo.unary_einsum")
    return std::make_unique<mhlo_unary_einsum>();
  if (name == "mhlo.uniform_dequantize")
    return std::make_unique<mhlo_uniform_dequantize>();
  if (name == "mhlo.uniform_quantize")
    return std::make_unique<mhlo_uniform_quantize>();
  if (name == "mhlo.while")
    return std::make_unique<mhlo_while>();
  if (name == "mhlo.xla.rng_get_and_update_state")
    return std::make_unique<mhlo_xla_rng_get_and_update_state>();
  if (name == "mhlo.xor")
    return std::make_unique<mhlo_xor>();
  assert(false && "Invalid op name");
}

} // namespace grammar
