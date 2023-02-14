/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Getters for Operation Infos                                                *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "OpInfos.h"

#include <cassert>
#include <memory>
#include <string>

class chlo_acos : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_acosh : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_asin : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_asinh : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_atan : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_atanh : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_bessel_i1e : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_add : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_and : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_atan2 : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_compare : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return CHLO_ComparisonDirectionAttr;
      case 2: return CHLO_ComparisonTypeAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_complex : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_divide : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_maximum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_minimum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_multiply : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_next_after : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_or : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_polygamma : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_power : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_remainder : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_select : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_left : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_right_arithmetic : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_right_logical : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_subtract : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_xor : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_zeta : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_conj : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_constant_like : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return TypedAttrInterface;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_constant : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_cosh : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_digamma : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_dynamic_reshape : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_erf : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_erfc : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_inf : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_neg_inf : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_pos_inf : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_lgamma : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_minimum_broadcast_shapes : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_526;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_next_after : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_polygamma : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_rank_specialization_cluster : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_rank_specialization_cluster_yield : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class chlo_sinh : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_tan : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_top_k : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_zeta : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_abs : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_632;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_add_dependency : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_add : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_after_all : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_all_gather : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
      case 1: return I64ElementsAttr;
      case 2: return ChannelHandle;
      case 3: return UnitAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_all_reduce : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return ChannelHandle;
      case 2: return UnitAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_all_to_all : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
      case 1: return I64Attr;
      case 2: return I64Attr;
      case 3: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_and : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_async_done : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return FlatSymbolRefAttr;
      case 1: return StrAttr;
      case 2: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_async_start : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return FlatSymbolRefAttr;
      case 1: return StrAttr;
      case 2: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_AsyncBundle;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_async_update : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return FlatSymbolRefAttr;
      case 1: return StrAttr;
      case 2: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_AsyncBundle;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_atan2 : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_batch_norm_grad : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return F32Attr;
      case 1: return I64Attr;
    }
    assert(false && "Invalid attribute index");
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

class mhlo_batch_norm_inference : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return F32Attr;
      case 1: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_batch_norm_training : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return F32Attr;
      case 1: return I64Attr;
    }
    assert(false && "Invalid attribute index");
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

class mhlo_bitcast_convert : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_bitcast : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_broadcast_in_dim : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_broadcast : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_case : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cbrt : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_ceil : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cholesky : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_clamp : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_count_leading_zeros : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_collective_permute : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return ChannelHandle;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_compare : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComparisonDirectionAttr;
      case 1: return HLO_ComparisonTypeAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_complex : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_compute_reshape_shape : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_754;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_concatenate : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_constant : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_convert : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_convolution : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
      case 3: return I64ElementsAttr;
      case 4: return BoolElementsAttr;
      case 5: return ConvDimensionNumbers;
      case 6: return I64Attr;
      case 7: return I64Attr;
      case 8: return HLO_PrecisionConfigAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_copy : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return UnitAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cosine : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_create_token : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cross_replica_sum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_cstr_reshapable : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return Shape_WitnessType;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_custom_call : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return StrAttr;
      case 1: return BoolAttr;
      case 2: return StrAttr;
      case 3: return HLO_CustomCallApiVersionAttr;
      case 4: return HLO_FlatSymbolRefArrayAttr;
      case 5: return HLO_ArrayOfLayoutAttr;
      case 6: return HLO_ArrayOfLayoutAttr;
      case 7: return anonymous_707;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_divide : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_domain : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_DomainKindAttr;
      case 1: return StrAttr;
      case 2: return StrAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dot_general : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return DotDimensionNumbers;
      case 1: return HLO_PrecisionConfigAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dot : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PrecisionConfigAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_broadcast_in_dim : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_conv : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
      case 3: return I64ElementsAttr;
      case 4: return BoolElementsAttr;
      case 5: return ConvDimensionNumbers;
      case 6: return I64Attr;
      case 7: return I64Attr;
      case 8: return HLO_PrecisionConfigAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_gather : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return GatherDimensionNumbers;
      case 1: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_iota : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_pad : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_reshape : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_slice : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_dynamic_update_slice : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_einsum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return StrAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_exponential : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_exponential_minus_one : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_fft : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FftTypeAttr;
      case 1: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_floor : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_fusion : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FusionKindAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_747;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_gather : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return GatherDimensionNumbers;
      case 1: return I64ElementsAttr;
      case 2: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_get_dimension_size : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return I32Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_get_tuple_element : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I32Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_if : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_imag : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_infeed : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return StrAttr;
      case 1: return ArrayAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_iota : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_is_finite : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_log_plus_one : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_log : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_logistic : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_map : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_maximum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_minimum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_multiply : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_negate : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_not : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_optimization_barrier : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_or : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_outfeed : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return StrAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_pad : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_partition_id : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_651;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_popcnt : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_power : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_real_dynamic_slice : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_real : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_recv : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ChannelHandle;
      case 1: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce_precision : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I32Attr;
      case 1: return I32Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce_scatter : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
      case 1: return I64ElementsAttr;
      case 2: return ChannelHandle;
      case 3: return UnitAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reduce_window : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
      case 3: return I64ElementsAttr;
      case 4: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_remainder : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_replica_id : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_651;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reshape : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_return : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_reverse : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_rng_bit_generator : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_RngAlgorithmAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntOrFpTensor;
      case 1: return HLO_IntOrFpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_rng : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_RngDistributionAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredIntOrFpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_round_nearest_even : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_round_nearest_afz : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_rsqrt : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_scatter : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ScatterDimensionNumbers;
      case 1: return BoolAttr;
      case 2: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_select_and_scatter : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_select : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_send : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ChannelHandle;
      case 1: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_set_dimension_size : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_shift_left : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_shift_right_arithmetic : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_shift_right_logical : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sign : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_632;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sine : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_slice : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
      case 1: return I64ElementsAttr;
      case 2: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sort : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
      case 1: return BoolAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_sqrt : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_stochastic_convert : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_subtract : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_tanh : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_torch_index_select : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
      case 1: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_trace : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return StrAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_transpose : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64ElementsAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_triangular_solve : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return BoolAttr;
      case 1: return BoolAttr;
      case 2: return BoolAttr;
      case 3: return HLO_TransposeAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_tuple : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tuple;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_unary_einsum : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return StrAttr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_uniform_dequantize : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_740;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_uniform_quantize : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_QuantizedIntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_while : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_xla_rng_get_and_update_state : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return I64Attr;
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_728;
    }
    assert(false && "Invalid result index");
  }
};

class mhlo_xor : public OpInfo {
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
  AttrType getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

std::string OpAndResTypeToString(OpAndResType type) {
  if (type == DefaultUnknownOpAndResType) return "DefaultUnknownOpAndResType";
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

std::string AttrTypeToString(AttrType type) {
  if (type == DefaultUnknownAttrType) return "DefaultUnknownAttrType";
  if (type == ArrayAttr) return "ArrayAttr";
  if (type == BoolAttr) return "BoolAttr";
  if (type == BoolElementsAttr) return "BoolElementsAttr";
  if (type == CHLO_ComparisonDirectionAttr) return "CHLO_ComparisonDirectionAttr";
  if (type == CHLO_ComparisonTypeAttr) return "CHLO_ComparisonTypeAttr";
  if (type == ChannelHandle) return "ChannelHandle";
  if (type == ConvDimensionNumbers) return "ConvDimensionNumbers";
  if (type == DotDimensionNumbers) return "DotDimensionNumbers";
  if (type == ElementsAttr) return "ElementsAttr";
  if (type == F32Attr) return "F32Attr";
  if (type == FlatSymbolRefAttr) return "FlatSymbolRefAttr";
  if (type == GatherDimensionNumbers) return "GatherDimensionNumbers";
  if (type == HLO_ArrayOfLayoutAttr) return "HLO_ArrayOfLayoutAttr";
  if (type == HLO_ComparisonDirectionAttr) return "HLO_ComparisonDirectionAttr";
  if (type == HLO_ComparisonTypeAttr) return "HLO_ComparisonTypeAttr";
  if (type == HLO_CustomCallApiVersionAttr) return "HLO_CustomCallApiVersionAttr";
  if (type == HLO_DomainKindAttr) return "HLO_DomainKindAttr";
  if (type == HLO_FftTypeAttr) return "HLO_FftTypeAttr";
  if (type == HLO_FlatSymbolRefArrayAttr) return "HLO_FlatSymbolRefArrayAttr";
  if (type == HLO_FusionKindAttr) return "HLO_FusionKindAttr";
  if (type == HLO_PrecisionConfigAttr) return "HLO_PrecisionConfigAttr";
  if (type == HLO_RngAlgorithmAttr) return "HLO_RngAlgorithmAttr";
  if (type == HLO_RngDistributionAttr) return "HLO_RngDistributionAttr";
  if (type == HLO_TransposeAttr) return "HLO_TransposeAttr";
  if (type == I32Attr) return "I32Attr";
  if (type == I64Attr) return "I64Attr";
  if (type == I64ElementsAttr) return "I64ElementsAttr";
  if (type == ScatterDimensionNumbers) return "ScatterDimensionNumbers";
  if (type == StrAttr) return "StrAttr";
  if (type == TypedAttrInterface) return "TypedAttrInterface";
  if (type == UnitAttr) return "UnitAttr";
  if (type == anonymous_707) return "anonymous_707";
  assert(false && "Invalid AttrType");
}

OpInfoPtr createOpInfo(std::string name) {
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

