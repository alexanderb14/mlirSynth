#include "CandidateStore.h"

using namespace llvm;
using namespace mlir;

void CandidateStore::addCandidate(const CandidatePtr &candidate) {
  std::lock_guard<std::mutex> lock(addCandidatesMutex);

  unsigned weight = candidate->getNumOps();
  IOType ioType = candidate->getIOType();

  candidateToId[candidate.get()] = candidateToId.size();

  if (weightToCandidates.find(weight) == weightToCandidates.end())
    weightToCandidates[weight] =
        std::unordered_map<IOType, std::vector<CandidatePtr>>();

  if (weightToCandidates[weight].find(ioType) ==
      weightToCandidates[weight].end())
    weightToCandidates[weight][ioType] = std::vector<CandidatePtr>();

  weightToCandidates[weight][ioType].push_back(candidate);
}

std::vector<CandidatePtr> CandidateStore::getCandidates() {
  std::vector<CandidatePtr> candidates;
  for (auto &weightToCandidate : weightToCandidates) {
    for (auto &ioTypeToCandidate : weightToCandidate.second) {
      for (auto &candidate : ioTypeToCandidate.second) {
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

std::vector<CandidatePtr> CandidateStore::getCandidates(unsigned weight) {
  std::vector<CandidatePtr> candidates;
  for (unsigned i = 0; i < weight; i++) {
    if (weightToCandidates.find(i) != weightToCandidates.end()) {
      for (auto &ioTypeToCandidate : weightToCandidates[i]) {
        for (auto &candidate : ioTypeToCandidate.second) {
          candidates.push_back(candidate);
        }
      }
    }
  }
  return candidates;
}

std::vector<CandidatePtr> CandidateStore::getCandidates(unsigned weight, IOType ioType) {
  std::vector<CandidatePtr> candidates;

  for (unsigned i = 0; i < weight; i++) {
    if (weightToCandidates.find(i) != weightToCandidates.end()) {
      if (weightToCandidates[i].find(ioType) != weightToCandidates[i].end()) {
        for (auto &candidate : weightToCandidates[i][ioType]) {
          candidates.push_back(candidate);
        }
      }
    }
  }
  return candidates;
}

void CandidateStore::merge(CandidateStorePtr &other) {
  for (auto &pair : other->weightToCandidates) {
    for (auto &ioTypeToCandidate : pair.second) {
      for (auto &candidate : ioTypeToCandidate.second) {
        addCandidate(candidate);
      }
    }
  }
}

void CandidateStore::dumpCandidates() {
  for (auto &pair : weightToCandidates) {
    llvm::outs() << "Weight: " << pair.first << "\n";
    for (auto &ioTypeToCandidate : pair.second) {
      llvm::outs() << "IOType: " << ioTypeToString(ioTypeToCandidate.first)
                   << "\n";
      for (auto &candidate : ioTypeToCandidate.second) {
        candidate->dump();
      }
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
