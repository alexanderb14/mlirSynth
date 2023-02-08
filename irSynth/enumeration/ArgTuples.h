#ifndef IRSYNTH_CANDIDATETUPLES_H
#define IRSYNTH_CANDIDATETUPLES_H

#include "Candidate.h"

struct ArgTuple {
  std::vector<CandidatePtr> operands;
  std::vector<mlir::Attribute> attributes;
  std::vector<std::shared_ptr<mlir::Region>> regions;
};

std::vector<ArgTuple>
getOperandArgTuples(mlir::MLIRContext &ctx,
                    mlir::RegisteredOperationName opName,
                    std::vector<CandidatePtr> &operandCandidates,
                    mlir::Block::BlockArgListType &functionArgs);

#endif // IRSYNTH_CANDIDATETUPLES_H
