#ifndef IRSYNTH_CANDIDATETUPLES_H
#define IRSYNTH_CANDIDATETUPLES_H

#include "Candidate.h"

struct ArgTuple {
  std::vector<CandidatePtr> operands;
  std::vector<mlir::Attribute> attributes;
  std::vector<std::shared_ptr<mlir::Region>> regions;
};

using RegionPtr = std::shared_ptr<mlir::Region>;

std::vector<ArgTuple>
getCartesianProduct(std::vector<std::vector<CandidatePtr>> &operands,
                     std::vector<std::vector<mlir::Attribute>> &attributes,
                     std::vector<std::vector<RegionPtr>> &regions);

#endif // IRSYNTH_CANDIDATETUPLES_H
