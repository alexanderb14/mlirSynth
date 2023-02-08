#ifndef IRSYNTH_STATS_H
#define IRSYNTH_STATS_H

#include <mutex>

class EnumerationStats {
public:
  void dump();
  void merge(EnumerationStats &other);

public:
  unsigned numEnumerated = 0;
  unsigned numValid = 0;
  unsigned numExecuted = 0;
  unsigned numIgnored = 0;
  unsigned numOps = 0;

private:
  std::mutex mutex;
};

#endif // IRSYNTH_STATS_H
