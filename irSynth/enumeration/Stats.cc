#include "Stats.h"

#include "llvm/Support/raw_ostream.h"

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

  llvm::outs() << "JSON: {"
               << "\"numEnumerated\":" << numEnumerated << ","
               << "\"numValid\":" << numValid << ","
               << "\"numExecuted\":" << numExecuted << ","
               << "\"numIgnored\":" << numIgnored << ","
               << "\"numOps\":" << numOps << "}\n";
}
