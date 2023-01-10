#ifndef TOOLS_ENUMERATOR_CANDIDATEENUMERATOR_H
#define TOOLS_ENUMERATOR_CANDIDATEENUMERATOR_H

#include "Candidate.h"

struct CandidateTuple {
  std::vector<CandidatePtr> operands;
  std::vector<mlir::Attribute> attributes;
  std::vector<std::shared_ptr<mlir::Region>> regions;
};

std::vector<CandidateTuple>
getOperandCandidateTuples(mlir::MLIRContext &ctx,
                          mlir::RegisteredOperationName opName,
                          std::vector<CandidatePtr> &operandCandidates);

#endif // TOOLS_ENUMERATOR_CANDIDATEENUMERATOR_H
