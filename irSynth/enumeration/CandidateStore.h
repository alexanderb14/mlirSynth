#ifndef IRSYNTH_CANDIDATESTORE_H
#define IRSYNTH_CANDIDATESTORE_H

#include "enumeration/Candidate.h"
#include "enumeration/Grammar.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include <mutex>

class CandidateStore;
using CandidateStorePtr = std::shared_ptr<CandidateStore>;

class CandidateStore {
public:
  void addCandidate(const CandidatePtr &candidate);
  std::vector<CandidatePtr> getCandidates();
  std::vector<CandidatePtr> getCandidates(unsigned weight);
  std::vector<CandidatePtr> getCandidates(unsigned weight, OpAndResType ioType);

  void merge(CandidateStorePtr &other);

  void dumpCandidates();
  void dumpSizes();
  int getTotal();

  int getCandidateId(const CandidatePtr &candidate);

  bool addCandidateHash(double hash);

private:
  std::mutex addCandidatesMutex;
  std::unordered_map<Candidate *, int> candidateToId;
  std::unordered_map<unsigned,
                     std::unordered_map<OpAndResType, std::vector<CandidatePtr>>>
      weightToCandidates;

  std::mutex hashesMutex;
  std::unordered_map<double, unsigned> hashes;
};

#endif // IRSYNTH_CANDIDATESTORE_H
