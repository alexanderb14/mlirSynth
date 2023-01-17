#ifndef EXPORTER_H
#define EXPORTER_H

#include "Scop.h"

#include <llvm/Support/raw_ostream.h>

void toDot(llvm::raw_ostream &os, Scop &scop);
void toDotStmts(llvm::raw_ostream &os, Scop &scop);

#endif // EXPORTER_H
