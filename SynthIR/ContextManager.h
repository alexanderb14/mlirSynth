#ifndef TOOLS_SYNTHESIZER_CONTEXTMANAGER_H
#define TOOLS_SYNTHESIZER_CONTEXTMANAGER_H

#include <mlir/IR/MLIRContext.h>

using MLIRContextPtr = std::shared_ptr<mlir::MLIRContext>;

class ContextManager {
public:
  ContextManager(bool printErrors);

  mlir::MLIRContext* createContext();

private:
  std::vector<MLIRContextPtr> contexts;

private:
  bool printErrors;
};
using ContextManagerPtr = std::shared_ptr<ContextManager>;

#endif // TOOLS_SYNTHESIZER_CONTEXTMANAGER_H
