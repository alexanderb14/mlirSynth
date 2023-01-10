#include "Candidate.h"

using namespace llvm;
using namespace mlir;

SmallVector<Value> Candidate::merge(MLIRContext &ctx,
                                    std::vector<CandidatePtr> &others) {
  // Merge other candidates into this one.
  BlockAndValueMapping mapping;
  mlir::DenseMap<unsigned, BlockArgument> seenArgs;
  for (auto &other : others) {
    auto &block = region->getBlocks().front();
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    // Merge other candidates arguments.
    unsigned argIdx = 0;
    for (auto &arg : other->region->getArguments()) {
      unsigned argId = other->argIds[argIdx++];
      argIds.push_back(argId);

      if (seenArgs.count(argId) == 0) {
        auto newArg = block.addArgument(arg.getType(), UnknownLoc::get(&ctx));
        seenArgs[argId] = newArg;
      }
      mapping.map(arg, seenArgs[argId]);
    }

    // Clone other candidates operations, and update the operands with the
    // mapping.
    for (auto &op : other->getRegion()->getBlocks().front()) {
      builder.insert(op.clone(mapping));
    }
  }

  // Get result values.
  SmallVector<Value> resultValues = {};
  for (auto &other : others) {
    for (auto result : other->getResults()) {
      resultValues.push_back(mapping.lookupOrDefault(result));
    }
  }

  // Add other candidates op counters to own.
  for (auto &other : others) {
    numOps += other->getNumOps();
  }

  return resultValues;
}

void Candidate::addArgument(MLIRContext &ctx, Type type, unsigned argId) {
  argIds.push_back(argId);

  auto &block = region->getBlocks().front();
  unsigned argIdx = block.getNumArguments();
  block.insertArgument(argIdx, type, UnknownLoc::get(&ctx));
}

void Candidate::addOperation(MLIRContext &ctx, Operation *op, bool count) {
  auto &block = region->getBlocks().front();
  OpBuilder builder(&ctx);
  builder.setInsertionPoint(&block, block.end());
  builder.insert(op);

  if (count)
    numOps++;
}

SmallVector<Value> Candidate::getResults() {
  if (region->empty() || region->front().empty()) {
    // Return arguments.
    auto args = region->getBlocks().front().getArguments();
    return SmallVector<Value>(args.begin(), args.end());
  }
  // Return result of last op.
  return region->getBlocks().front().back().getResults();
}

void Candidate::dump() {
  llvm::outs() << "Candidate: \n";
  llvm::outs() << "- Args\n";
  unsigned argIdx = 0;
  for (auto &argId : argIds) {
    llvm::outs() << "  argId: " << argId << ", "
      << getRegion()->getBlocks().front().getArgument(argIdx) << "\n";
  }

  llvm::outs() << "- Ops\n";
  for (auto &op : region->getBlocks().front()) {
    llvm::outs() << "  ";
    op.dump();
  }
}

// CandidateStore
// -----------------------------------
void CandidateStore::addCandidate(const CandidatePtr &candidate, unsigned weight) {
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
