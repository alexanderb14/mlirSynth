#ifndef IRSYNTH_STATS_H
#define IRSYNTH_STATS_H

#include "ProcessingStatus.h"

#include <mutex>
#include <unordered_map>

class SynthesisStats {
public:
  SynthesisStats();

  void addProcessingStatus(ProcessingStatus status);
  void merge(SynthesisStats &other);

  void dump();

public:
  unsigned numEnumerated = 0;
  unsigned numValid = 0;
  unsigned numExecuted = 0;
  unsigned numIgnored = 0;
  unsigned numOps = 0;

private:
  std::mutex mutex;
  std::unordered_map<ProcessingStatus, unsigned> processingStatusCounts;
};

#endif // IRSYNTH_STATS_H
