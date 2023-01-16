#include "AttributeGen.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <random>

using namespace mlir;

int randomInteger(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  return dis(gen);
}

std::vector<Attribute>
getTensorAttributes(OpBuilder &builder, Region::BlockArgListType &functionArgs,
                    int maxRank) {
  std::vector<Attribute> tensorValues;

  auto attr = std::vector<Attribute>();
  attr.push_back(builder.getI64IntegerAttr(5));
  attr.push_back(builder.getI64IntegerAttr(3));
  attr.push_back(builder.getI64IntegerAttr(1));
  attr.push_back(builder.getI64IntegerAttr(7));
  Type type = RankedTensorType::get({static_cast<long>(attr.size())},
                                    attr[0].cast<TypedAttr>().getType());
  auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
  tensorValues.push_back(attrDense);


  if (maxRank >= 0) {
    std::vector<Attribute> attrs = {
        //        builder.getBoolAttr(true),    builder.getBoolAttr(false),
        builder.getF64FloatAttr(0.0),
        builder.getF64FloatAttr(1.0),
        //        builder.getI64IntegerAttr(0), builder.getI64IntegerAttr(1),
        IntegerAttr::get(
            IntegerType::get(builder.getContext(), 64, IntegerType::Signless),
            APInt(64, 0, false)),
    };
    for (auto attr : attrs) {
      Type type = RankedTensorType::get({}, attr.cast<TypedAttr>().getType());
      auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
      tensorValues.push_back(attrDense);
    }
  }

  if (maxRank >= 1) {
    std::vector<std::vector<Attribute>> attrs = {
        std::vector<Attribute>{builder.getF64FloatAttr(0)},
        std::vector<Attribute>{builder.getF64FloatAttr(1)}};
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        attrs.push_back(std::vector<Attribute>{builder.getF64FloatAttr(i),
                                               builder.getF64FloatAttr(j)});
      }
    }
    for (auto attr : attrs) {
      Type type = RankedTensorType::get({static_cast<long>(attr.size())},
                                        attr[0].cast<TypedAttr>().getType());
      auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
      tensorValues.push_back(attrDense);
    }
  }

  if (maxRank >= 2) {
    int n = 2;
    std::vector<Attribute> attr;
    attr.reserve(n * n);
    for (int i = 0; i < n * n; i++) {
      attr.push_back(builder.getF64FloatAttr(randomInteger(0, 10)));
    }

    std::vector<std::vector<Attribute>> attrs;
    attrs.push_back(attr);

    for (auto attr : attrs) {
      Type type =
          RankedTensorType::get({n, n}, attr[0].cast<TypedAttr>().getType());
      auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attr);
      tensorValues.push_back(attrDense);
    }
  }

  return tensorValues;
}
