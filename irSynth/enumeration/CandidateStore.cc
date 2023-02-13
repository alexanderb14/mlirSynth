#include "CandidateStore.h"

using namespace llvm;
using namespace mlir;

void CandidateStore::addCandidate(const CandidatePtr &candidate,
                                  unsigned weight) {
  std::lock_guard<std::mutex> lock(addCandidatesMutex);

  candidateToId[candidate.get()] = candidateToId.size();

  if (weightToCandidates.find(weight) == weightToCandidates.end())
    weightToCandidates[weight] = std::vector<CandidatePtr>();

  weightToCandidates[weight].push_back(candidate);
}

std::vector<CandidatePtr> CandidateStore::getCandidates() {
  std::vector<CandidatePtr> candidates;
  for (auto &weightToCandidate : weightToCandidates) {
    for (auto &candidate : weightToCandidate.second) {
      candidates.push_back(candidate);
    }
  }
  return candidates;
}

std::vector<CandidatePtr> CandidateStore::getCandidates(unsigned weight) {
  std::vector<CandidatePtr> candidates;
  for (unsigned i = 0; i < weight; i++) {
    if (weightToCandidates.find(i) != weightToCandidates.end()) {
      for (auto &candidate : weightToCandidates[i]) {
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

void CandidateStore::merge(CandidateStorePtr &other) {
  for (auto &pair : other->weightToCandidates) {
    for (auto &candidate : pair.second) {
      addCandidate(candidate, pair.first);
    }
  }
}

void CandidateStore::dumpCandidates() {
  for (auto &pair : weightToCandidates) {
    llvm::outs() << "Weight: " << pair.first << "\n";
    for (auto &candidate : pair.second) {
      candidate->dump();
    }
  }
}

void CandidateStore::dumpSizes() {
  llvm::outs() << "\nCandidateStore contents (length: number of candidates)"
               << "\n--------\n";
  for (auto &pair : weightToCandidates) {
    llvm::errs() << pair.first << ": " << pair.second.size() << "\n";
  }
}

int CandidateStore::getTotal() {
  int numCandidates = 0;
  for (auto &pair : weightToCandidates) {
    numCandidates += pair.second.size();
  }
  return numCandidates;
}

int CandidateStore::getCandidateId(const CandidatePtr &candidate) {
  auto pos = candidateToId.find(candidate.get());
  if (pos == candidateToId.end())
    return -1;
  return pos->second;
}

bool CandidateStore::addCandidateHash(double hash) {
  std::lock_guard<std::mutex> lock(hashesMutex);

  if (hashes.find(hash) == hashes.end()) {
    hashes[hash] = 1;
    return true;
  }

  hashes[hash]++;
  return false;
}
