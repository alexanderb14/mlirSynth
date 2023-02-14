/*===- Cartesian Product Generators -----------------------------*- C++ -*-===*\
|*                                                                            *|
|* The arguments to ranges::views::cartesian_product need to be known         *|
|* at compile time. Therefore, this generator-approach. TODO: Replace.        *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "ArgTuples.h"

#include "enumeration/Generators.h"

#include <range/v3/all.hpp>
#include <range/v3/view/cartesian_product.hpp>

#include <iostream>
#include <vector>

using namespace llvm;
using namespace mlir;

std::vector<ArgTuple>
getCartesianProduct(std::vector<std::vector<CandidatePtr>> &operands,
                    std::vector<std::vector<Attribute>> &attributes,
                    std::vector<std::vector<RegionPtr>> &regions) {
  unsigned numOperands = operands.size();
  unsigned numAttributes = attributes.size();
  unsigned numRegions = regions.size();

  std::vector<ArgTuple> ret;
  if (numOperands == 0 && numAttributes == 0 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {};
      tuple.regions = {std::get<0>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 0 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {};
      tuple.regions = {std::get<0>(cand), std::get<1>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 1 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(attributes[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {std::get<0>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 1 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(attributes[0], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {std::get<0>(cand)};
      tuple.regions = {std::get<1>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 1 && numRegions == 2) {
    auto cands =
        ranges::views::cartesian_product(attributes[0], regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {std::get<0>(cand)};
      tuple.regions = {std::get<1>(cand), std::get<2>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 2 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(attributes[0], attributes[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {std::get<0>(cand), std::get<1>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 2 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(attributes[0], attributes[1],
                                                  regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {std::get<0>(cand), std::get<1>(cand)};
      tuple.regions = {std::get<2>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 0 && numAttributes == 2 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(attributes[0], attributes[1],
                                                  regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {};
      tuple.attributes = {std::get<0>(cand), std::get<1>(cand)};
      tuple.regions = {std::get<2>(cand), std::get<3>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 0 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 0 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(operands[0], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {};
      tuple.regions = {std::get<1>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 0 && numRegions == 2) {
    auto cands =
        ranges::views::cartesian_product(operands[0], regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {};
      tuple.regions = {std::get<1>(cand), std::get<2>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 1 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0], attributes[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {std::get<1>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 1 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(operands[0], attributes[0],
                                                  regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {std::get<1>(cand)};
      tuple.regions = {std::get<2>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 1 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(operands[0], attributes[0],
                                                  regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {std::get<1>(cand)};
      tuple.regions = {std::get<2>(cand), std::get<3>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 2 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0], attributes[0],
                                                  attributes[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {std::get<1>(cand), std::get<2>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 2 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(operands[0], attributes[0],
                                                  attributes[1], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {std::get<1>(cand), std::get<2>(cand)};
      tuple.regions = {std::get<3>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 1 && numAttributes == 2 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(
        operands[0], attributes[0], attributes[1], regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand)};
      tuple.attributes = {std::get<1>(cand), std::get<2>(cand)};
      tuple.regions = {std::get<3>(cand), std::get<4>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 0 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 0 && numRegions == 1) {
    auto cands =
        ranges::views::cartesian_product(operands[0], operands[1], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {};
      tuple.regions = {std::get<2>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 0 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {};
      tuple.regions = {std::get<2>(cand), std::get<3>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 1 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  attributes[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {std::get<2>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 1 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  attributes[0], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {std::get<2>(cand)};
      tuple.regions = {std::get<3>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 1 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(
        operands[0], operands[1], attributes[0], regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {std::get<2>(cand)};
      tuple.regions = {std::get<3>(cand), std::get<4>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 2 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  attributes[0], attributes[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {std::get<2>(cand), std::get<3>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 2 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(
        operands[0], operands[1], attributes[0], attributes[1], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {std::get<2>(cand), std::get<3>(cand)};
      tuple.regions = {std::get<4>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 2 && numAttributes == 2 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  attributes[0], attributes[1],
                                                  regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
      tuple.attributes = {std::get<2>(cand), std::get<3>(cand)};
      tuple.regions = {std::get<4>(cand), std::get<5>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 0 && numRegions == 0) {
    auto cands =
        ranges::views::cartesian_product(operands[0], operands[1], operands[2]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 0 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  operands[2], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {};
      tuple.regions = {std::get<3>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 0 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(
        operands[0], operands[1], operands[2], regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {};
      tuple.regions = {std::get<3>(cand), std::get<4>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 1 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  operands[2], attributes[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {std::get<3>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 1 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(
        operands[0], operands[1], operands[2], attributes[0], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {std::get<3>(cand)};
      tuple.regions = {std::get<4>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 1 && numRegions == 2) {
    auto cands =
        ranges::views::cartesian_product(operands[0], operands[1], operands[2],
                                         attributes[0], regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {std::get<3>(cand)};
      tuple.regions = {std::get<4>(cand), std::get<5>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 2 && numRegions == 0) {
    auto cands = ranges::views::cartesian_product(
        operands[0], operands[1], operands[2], attributes[0], attributes[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {std::get<3>(cand), std::get<4>(cand)};
      tuple.regions = {};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 2 && numRegions == 1) {
    auto cands = ranges::views::cartesian_product(operands[0], operands[1],
                                                  operands[2], attributes[0],
                                                  attributes[1], regions[0]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {std::get<3>(cand), std::get<4>(cand)};
      tuple.regions = {std::get<5>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }
  if (numOperands == 3 && numAttributes == 2 && numRegions == 2) {
    auto cands = ranges::views::cartesian_product(
        operands[0], operands[1], operands[2], attributes[0], attributes[1],
        regions[0], regions[1]);
    for (auto cand : cands) {
      ArgTuple tuple;
      tuple.operands = {std::get<0>(cand), std::get<1>(cand),
                        std::get<2>(cand)};
      tuple.attributes = {std::get<3>(cand), std::get<4>(cand)};
      tuple.regions = {std::get<5>(cand), std::get<6>(cand)};
      ret.push_back(tuple);
    }
    return ret;
  }

  llvm::outs() << "Unsupported number of operands (" << numOperands
               << "), attributes (" << numAttributes << "), regions ("
               << numRegions << ")\n";
  assert(false);
}
