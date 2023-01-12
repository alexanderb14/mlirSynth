#ifndef IRSYNTH_STATS_H
#define IRSYNTH_STATS_H

class EnumerationStats {
public:
  void dump();

public:
  unsigned numEnumerated = 0;
  unsigned numValid = 0;
  unsigned numExecuted = 0;
  unsigned numIgnored = 0;
  unsigned numOps = 0;
};

#endif // IRSYNTH_STATS_H
