#ifndef IRSYNTH_CANDIDATE_H
#define IRSYNTH_CANDIDATE_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include <mutex>

class Candidate;
using CandidatePtr = std::shared_ptr<Candidate>;

class Candidate {
public:
  Candidate(std::vector<CandidatePtr> predecessors) {
    this->predecessors = predecessors;
    region = std::make_shared<mlir::Region>();
    region->push_back(new mlir::Block);
  }

  mlir::SmallVector<mlir::Value> merge(mlir::MLIRContext &ctx,
                                       std::vector<CandidatePtr> &others);

  void addArgument(mlir::MLIRContext &ctx, mlir::Type type, unsigned argId);
  void addOperation(mlir::MLIRContext &ctx, mlir::Operation *op,
                    bool count = true);

  mlir::Operation *getBegin() { return &region->getBlocks().front().front(); }
  mlir::Region *getRegion() { return region.get(); }
  std::vector<unsigned> getArgIds() { return argIds; }

  mlir::Operation *getEnd() { return &this->getBlock()->back(); }
  mlir::Block *getBlock() { return &region->getBlocks().front(); }

  mlir::SmallVector<mlir::Value> getResults();
  int getNumOps() { return numOps; }
  int getNumArguments() {
    return region->getBlocks().front().getNumArguments();
  }
  void dump();

  std::vector<CandidatePtr> getPredecessors() { return predecessors; }

  bool hasHash() { return hashExists; }
  void setHash(double hash) {
    hashExists = true;
    this->hash = hash;
  }
  double getHash() { return hash; }

private:
  std::vector<unsigned> argIds;
  std::shared_ptr<mlir::Region> region;
  int numOps = 0;

  std::vector<CandidatePtr> predecessors;

  double hash;
  bool hashExists = false;
};

class CandidateStore;
using CandidateStorePtr = std::shared_ptr<CandidateStore>;

class CandidateStore {
public:
  void addCandidate(const CandidatePtr &candidate, unsigned weight);
  std::vector<CandidatePtr> getCandidates();
  std::vector<CandidatePtr> getCandidates(unsigned weight);

  void merge(CandidateStorePtr &other);

  void dumpCandidates();
  void dumpSizes();
  int getTotal();

  int getCandidateId(const CandidatePtr &candidate);

  bool addCandidateHash(double hash);

private:
  std::mutex addCandidatesMutex;
  std::unordered_map<Candidate *, int> candidateToId;
  std::unordered_map<unsigned, std::vector<CandidatePtr>> weightToCandidates;

  std::mutex hashesMutex;
  std::unordered_map<double, unsigned> hashes;
};

#endif // IRSYNTH_CANDIDATE_H
