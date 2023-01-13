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

template <int n_op, int n_attr, int n_region>
auto cartesianProduct(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) -> std::vector<CandidateTuple>;

template<int n_op, int n_attr, int n_region>
struct CartesianProduct;

template<>
struct CartesianProduct<0, 0, 0> {
  template<class ... Args>
  static auto apply(std::vector<CandidatePtr> &/*operandCandidates*/,
                    std::vector<Attribute> &/*attributeCandidates*/,
                    std::vector<std::shared_ptr<Region>> &/*regionCandidates*/,
                    Args && ... args) -> std::vector<CandidateTuple> {
    return ranges::views::cartesian_product(std::forward<Args>(args)...);
  }
};



template<int n_region>
struct CartesianProduct<0, 0, n_region> {
  template<class ... Args>
  static auto apply(std::vector<CandidatePtr> &operandCandidates,
                    std::vector<Attribute> &attributeCandidates,
                    std::vector<std::shared_ptr<Region>> &regionCandidates,
                    Args && ... args) -> std::vector<CandidateTuple> {
  return CartesianProduct<0, 0, n_region - 1>::apply(operandCandidates, attributeCandidates, regionCandidates, std::forward<Args>(args)..., regionCandidates);
}
};


template<int n_attr, int n_region>
struct CartesianProduct<0, n_attr, n_region> {
  template<class ... Args>
  static auto apply(std::vector<CandidatePtr> &operandCandidates,
                    std::vector<Attribute> &attributeCandidates,
                    std::vector<std::shared_ptr<Region>> &regionCandidates,
                    Args && ... args) -> std::vector<CandidateTuple> {
  return CartesianProduct<0, n_attr - 1, n_region>::apply(operandCandidates, attributeCandidates, regionCandidates, std::forward<Args>(args)..., attributeCandidates);
}
};


template<int n_op, int n_attr, int n_region>
struct CartesianProduct {
  template<class ... Args>
  static auto apply(std::vector<CandidatePtr> &operandCandidates,
                    std::vector<Attribute> &attributeCandidates,
                    std::vector<std::shared_ptr<Region>> &regionCandidates,
                    Args && ... args) -> std::vector<CandidateTuple> {
  return CartesianProduct<n_op - 1, n_attr, n_region>::apply(operandCandidates, attributeCandidates, regionCandidates, std::forward<Args>(args)..., operandCandidates);
}
};

template<class U, class U2, int start, class T, T ... ints>
auto createTupleImpl(U2 & cand, std::integer_sequence<T, ints...> int_seq) -> U2 {
  return {std::get<start + ints>(cand)...};
}

template<class T, class U2, int start, std::size_t N, typename Indices = std::make_index_sequence<N>>
auto createTuple(U2 & cand) -> T {
  return createTupleImpl<start>(cand, Indices{});
}

template <int n_op, int n_attr, int n_region>
std::vector<CandidateTuple>
getCartesianProduct(std::vector<CandidatePtr> &operandCandidates,
                       std::vector<Attribute> &attributeCandidates,
                       std::vector<std::shared_ptr<Region>> &regionCandidates) {
  auto cands = CartesianProduct<n_op, n_attr, n_region>::apply(operandCandidates, attributeCandidates, regionCandidates);
  std::vector<CandidateTuple> ret;
  for (auto cand : cands) {
    CandidateTuple tuple;
    //tuple.operands = {std::get<0>(cand)};
    tuple.operands = createTuple<decltype(tuple.operands), 0, n_op>(cand);
    if constexpr (n_attr > 0) {
      tuple.attributes = createTuple<decltype(tuple.attributes), n_op, n_attr>(cand);
    }
    if constexpr (n_region > 0) {
      tuple.regions = createTuple<decltype(tuple.regions), n_op + n_attr, n_region>(cand);
    }
    ret.push_back(tuple);
  }
  return ret;
}

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

//static constexpr std::array<std::array>>> foo = createArray(2,2,2);
//constexpr auto createArray(int x, int y, int z) {
//  for (int i = 0; i < x; i++) {
//    for (int j = 0; j < y; j++) {
//      for (int k = 0; k < z; k++) {
//        foo[i,j,k] = getCartesianProduct<i,j,k>;
//      }
//    }
//  }
//}

std::vector<CandidateTuple>
getOperandCandidateTuples(MLIRContext &ctx, RegisteredOperationName opName,
                          std::vector<CandidatePtr> &operandCandidates) {
  OpBuilder builder(&ctx);
  Operation *op =
      builder.create(UnknownLoc::get(&ctx), opName.getIdentifier(), {});

  int numOperands = getRequiredNumOperands(op);
  int numAttributes = getRequiredNumAttributes(op);
  std::vector<Attribute> attributeCandidates = getTensorAttributes(builder, 2);
  int numRegions = getRequiredNumRegions(op);
  std::vector<std::shared_ptr<Region>> regionCandidates = getRegions(builder);

//  return foo[numOperands-1][numAttributes-1][numRegions-1]()

  if (numOperands == 1 && numAttributes == 1 && numRegions == 0)
    return getCartesianProduct<1, 1, 0> (operandCandidates, attributeCandidates, regionCandidates);
    //return getCartesianProduct110(operandCandidates, attributeCandidates,
    //                              regionCandidates);
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
