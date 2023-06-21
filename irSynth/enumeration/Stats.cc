#include "Stats.h"

#include "llvm/Support/raw_ostream.h"

EnumerationStats::EnumerationStats() {
  // Initialize all the processing status counts with 0.
  for (int i = 0; i < static_cast<int>(ProcessingStatus::ENUM_SIZE); i++) {
    processingStatusCounts.insert({(ProcessingStatus)i, 0});
  }
}

void EnumerationStats::addProcessingStatus(ProcessingStatus status) {
  processingStatusCounts[status]++;
}

void EnumerationStats::merge(EnumerationStats &other) {
  std::lock_guard<std::mutex> lock(mutex);

  numEnumerated += other.numEnumerated;
  numValid += other.numValid;
  numExecuted += other.numExecuted;
  numIgnored += other.numIgnored;
  numOps += other.numOps;

  for (auto &kv : other.processingStatusCounts) {
    processingStatusCounts[kv.first] += kv.second;
  }
}

void EnumerationStats::dump() {
  llvm::outs() << "Enumeration Stats"
               << "\n--------\n";
  llvm::outs() << "Number of enumerated candidates:             "
               << numEnumerated << "\n";

  llvm::outs() << "Number of valid candidates:                  " << numValid
               << "\n";
  llvm::outs() << "Percentage of valid candidates:              "
               << (numValid * 100.0) / numEnumerated << "%\n";

  llvm::outs() << "Number of executed candidates:               " << numExecuted
               << "\n";
  llvm::outs() << "Percentage of executed candidates:           "
               << (numExecuted * 100.0) / numEnumerated << "%\n";

  llvm::outs() << "Number of ignored equivalent candidates:     " << numIgnored
               << "\n";

  llvm::outs() << "Number of operations in solution candidate:  " << numOps
               << "\n";

  llvm::outs() << "Processing Statuses:\n";
  for (auto &kv : processingStatusCounts) {
    ProcessingStatus status = kv.first;
    llvm::outs() << "  " << processingStatusToStr(status) << ": " << kv.second
                 << "\n";
  }

  llvm::outs() << "JSON: {";

  llvm::outs() << "\"numEnumerated\":" << numEnumerated << ","
               << "\"numValid\":" << numValid << ","
               << "\"numExecuted\":" << numExecuted << ","
               << "\"numIgnored\":" << numIgnored << ","
               << "\"numOps\":" << numOps << ",";

  llvm::outs() << "\"processingStatusCounts\":{";
  bool first = true;
  for (auto &kv : processingStatusCounts) {
    if (!first) {
      llvm::outs() << ",";
    }
    first = false;
    ProcessingStatus status = kv.first;
    llvm::outs() << "\"" << processingStatusToStr(status) << "\":" << kv.second;
  }
  llvm::outs() << "}";

  llvm::outs() << "}\n";
}
