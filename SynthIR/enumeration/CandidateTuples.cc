#include "CandidateTuples.h"

#include "Utils.h"

#include <range/v3/all.hpp>
#include <range/v3/view/cartesian_product.hpp>

#include <iostream>
#include <vector>

using namespace llvm;
using namespace mlir;

// FIXME: This is a hack to get the cartesian product for fixed size of
// arguments that need to be known at compile-time. Therefore, specialized
// functions are created for some sizes.

std::vector<CandidateTuple>
getCartesianProduct110(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, attributeCandidates);
  std::vector<CandidateTuple> ret;
  for (auto cand : cands) {
    CandidateTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<CandidateTuple>
getCartesianProduct111(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, attributeCandidates, regionCandidates);
  std::vector<CandidateTuple> ret;
  for (auto cand : cands) {
    CandidateTuple tuple;
    tuple.operands = {std::get<0>(cand)};
    tuple.attributes = {std::get<1>(cand)};
    tuple.regions = {std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<CandidateTuple>
getCartesianProduct211(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, operandCandidates,
                                       attributeCandidates, regionCandidates);
  std::vector<CandidateTuple> ret;
  for (auto cand : cands) {
    CandidateTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand)};
    tuple.regions = {std::get<3>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<CandidateTuple>
getCartesianProduct200(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands =
      ranges::views::cartesian_product(operandCandidates, operandCandidates);
  std::vector<CandidateTuple> ret;
  for (auto cand : cands) {
    CandidateTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<CandidateTuple>
getCartesianProduct210(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = ranges::views::cartesian_product(
      operandCandidates, operandCandidates, attributeCandidates);
  std::vector<CandidateTuple> ret;
  for (auto cand : cands) {
    CandidateTuple tuple;
    tuple.operands = {std::get<0>(cand), std::get<1>(cand)};
    tuple.attributes = {std::get<2>(cand)};
    ret.push_back(tuple);
  }
  return ret;
}

std::vector<CandidateTuple>
getOperandCandidateTuples(MLIRContext &ctx, RegisteredOperationName opName,
                          std::vector<CandidatePtr> &operandCandidates) {
  OpBuilder builder(&ctx);
  Operation *op =
      builder.create(UnknownLoc::get(&ctx), opName.getIdentifier(), {});

  int numOperands = getRequiredNumOperands(op);
  int numAttributes = getRequiredNumAttributes(op);
  std::vector<Attribute> attributeCandidates = getTensorAttributes(builder);
  int numRegions = getRequiredNumRegions(op);
  std::vector<std::shared_ptr<Region>> regionCandidates = getRegions(builder);

  if (numOperands == 1 && numAttributes == 1 && numRegions == 0)
    return getCartesianProduct110(operandCandidates, attributeCandidates,
                                  regionCandidates);
  if (numOperands == 1 && numAttributes == 1 && numRegions == 1)
    return getCartesianProduct111(operandCandidates, attributeCandidates,
                                  regionCandidates);
  if (numOperands == 2 && numAttributes == 0 && numRegions == 0)
    return getCartesianProduct200(operandCandidates, attributeCandidates,
                                  regionCandidates);
  if (numOperands == 2 && numAttributes == 1 && numRegions == 0)
    return getCartesianProduct210(operandCandidates, attributeCandidates,
                                  regionCandidates);
  if (numOperands == 2 && numAttributes == 1 && numRegions == 1)
    return getCartesianProduct211(operandCandidates, attributeCandidates,
                                  regionCandidates);

  llvm::outs() << "Unsupported number of operands (" << numOperands
               << "), attributes (" << numAttributes << "), regions ("
               << numRegions << ") in op: " << opName.getIdentifier() << "\n";
  assert(false);
}
