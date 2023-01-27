// WARNING: DO NOT EDIT THIS FILE. IT IS AUTOGENERATED BY
// scripts/gen_ArgTuples.py

#include "ArgTuples.h"
#include "AttributeGen.h"

#include "Utils.h"

#include <range/v3/all.hpp>
#include <range/v3/view/cartesian_product.hpp>

#include <iostream>
#include <vector>

using namespace llvm;
using namespace mlir;

std::vector<ArgTuple> get0operands0attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.regions = {std::get<0>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands0attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.regions = {std::get<0>(cand), std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands1attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(attributeCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.attributes = {std::get<0>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands1attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(attributeCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.attributes = {std::get<0>(cand)};
    tuple.regions = {std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands1attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      attributeCandidates, regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.attributes = {std::get<0>(cand)};
    tuple.regions = {std::get<1>(cand), std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands2attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(attributeCandidates,
                                                attributeCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.attributes = {std::get<0>(cand), std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands2attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      attributeCandidates, attributeCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.attributes = {std::get<0>(cand), std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get0operands2attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(attributeCandidates, attributeCandidates,
                                       regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.attributes = {std::get<0>(cand), std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand), std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands0attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(operandCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands0attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.regions = {std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands0attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.regions = {std::get<1>(cand), std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands1attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, attributeCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands1attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, attributeCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands1attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, attributeCandidates,
                                       regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand), std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands2attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, attributeCandidates, attributeCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand), std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands2attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, attributeCandidates,
                                       attributeCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand), std::get<2>(cand)};
    tuple.regions = {std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get1operands2attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, attributeCandidates, attributeCandidates,
      regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand), std::get<2>(cand)};
    tuple.regions = {std::get<3>(cand), std::get<4>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands0attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, operandCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands0attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands0attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand), std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands1attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, attributeCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands1attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, operandCandidates,
                                       attributeCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand)};
    tuple.regions = {std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands1attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, attributeCandidates,
      regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand)};
    tuple.regions = {std::get<3>(cand), std::get<4>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands2attributes0regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, attributeCandidates,
      attributeCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand), std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands2attributes1regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, attributeCandidates,
      attributeCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand), std::get<3>(cand)};
    tuple.regions = {std::get<4>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple> get2operands2attributes2regions(
    std::vector<CandidatePtr> &operandCandidates,
    std::vector<Attribute> &attributeCandidates,
    std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, attributeCandidates,
      attributeCandidates, regionCandidates, regionCandidates);
  std::vector<ArgTuple> ret;
  for (auto cand : cands) {
    ArgTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand), std::get<3>(cand)};
    tuple.regions = {std::get<4>(cand), std::get<5>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<ArgTuple>
getOperandArgTuples(MLIRContext &ctx, RegisteredOperationName opName,
                    std::vector<CandidatePtr> &operandCandidates) {
  OpBuilder builder(&ctx);
  Operation *op =
      builder.create(UnknownLoc::get(&ctx), opName.getIdentifier(), {});

  int numOperands = getRequiredNumOperands(op);
  int numAttributes = getRequiredNumAttributes(op);

  Block::BlockArgListType blockArgs;
  std::vector<Attribute> attributeCandidates =
      genAttributes(builder, blockArgs, 2);
  int numRegions = getRequiredNumRegions(op);

  std::vector<std::shared_ptr<Region>> regionCandidates = genRegions(builder);
  if (numOperands == 0 && numAttributes == 0 && numRegions == 1) {
    return get0operands0attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 0 && numRegions == 2) {
    return get0operands0attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 1 && numRegions == 0) {
    return get0operands1attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 1 && numRegions == 1) {
    return get0operands1attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 1 && numRegions == 2) {
    return get0operands1attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 2 && numRegions == 0) {
    return get0operands2attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 2 && numRegions == 1) {
    return get0operands2attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 0 && numAttributes == 2 && numRegions == 2) {
    return get0operands2attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 0 && numRegions == 0) {
    return get1operands0attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 0 && numRegions == 1) {
    return get1operands0attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 0 && numRegions == 2) {
    return get1operands0attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 1 && numRegions == 0) {
    return get1operands1attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 1 && numRegions == 1) {
    return get1operands1attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 1 && numRegions == 2) {
    return get1operands1attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 2 && numRegions == 0) {
    return get1operands2attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 2 && numRegions == 1) {
    return get1operands2attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 1 && numAttributes == 2 && numRegions == 2) {
    return get1operands2attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 0 && numRegions == 0) {
    return get2operands0attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 0 && numRegions == 1) {
    return get2operands0attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 0 && numRegions == 2) {
    return get2operands0attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 1 && numRegions == 0) {
    return get2operands1attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 1 && numRegions == 1) {
    return get2operands1attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 1 && numRegions == 2) {
    return get2operands1attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 2 && numRegions == 0) {
    return get2operands2attributes0regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 2 && numRegions == 1) {
    return get2operands2attributes1regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }
  if (numOperands == 2 && numAttributes == 2 && numRegions == 2) {
    return get2operands2attributes2regions(
        operandCandidates, attributeCandidates, regionCandidates);
  }

  llvm::outs() << "Unsupported number of operands (" << numOperands
               << "), attributes (" << numAttributes << "), regions ("
               << numRegions << ") in op: " << opName.getIdentifier() << "\n";
  assert(false);
}
