#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include "Scop.h"
#include "isl/isl_tools.h"

#include <regex>
#include <string>

using namespace llvm;
using namespace mlir;

// Create an isl_multi_union_aff that defines an identity mapping from the
// elements of USet to their N-th dimension.
//
// # Example: Domain: { A[i,j]; B[i,j,k] }, N: 1
// Resulting Mapping: { {A[i,j] -> [(j)]; B[i,j,k] -> [(j)] }
isl::multi_union_pw_aff mapToDimension(const isl::union_set& uSet, unsigned n) {
  // llvm::outs() << "mapToDimension " << stringFromIslObj(USet) << " " << N << "\n";
  assert(!uSet.is_null());
  assert(!uSet.is_empty());

  auto result = isl::union_pw_multi_aff::empty(uSet.get_space());

  for (isl::set s : uSet.get_set_list()) {
    unsigned dim = unsignedFromIslSize(s.tuple_dim());
    assert(dim >= n);
    auto pma = isl::pw_multi_aff::project_out_map(s.get_space(), isl::dim::set,
                                                  n, dim - n);
    if (n > 1)
      pma = pma.drop_dims(isl::dim::out, 0, n - 1);

    result = result.add_pw_multi_aff(pma);
  }

  // llvm::outs() << "MUPA: " << stringFromIslObj(isl::multi_union_pw_aff(isl::union_pw_multi_aff(Result))) << "\n";
  return isl::multi_union_pw_aff(isl::union_pw_multi_aff(result));
}

isl::schedule combineInSequence(const isl::schedule& prev, const isl::schedule& succ) {
  if (prev.is_null())
    return succ;
  if (succ.is_null())
    return prev;

  return prev.sequence(succ);
}

mlir::DenseMap<ScopStmt*, bool> seenStmts;

isl::schedule scheduleBlock(Block &block, Scop &scop, unsigned depth);
isl::schedule scheduleRegion(Region &region, Scop &scop, unsigned depth);
isl::schedule scheduleOperation(Operation *op, Scop &scop, unsigned depth);

isl::schedule scheduleBlock(Block &block, Scop &scop, unsigned depth) {
  isl::schedule sched;
  for (Operation &op : block.getOperations()) {
    unsigned depthNew = depth;
    if (isa<AffineForOp>(op))
      depthNew++;

    sched = combineInSequence(sched, scheduleOperation(&op, scop, depthNew));
  }

  return sched;
}

isl::schedule scheduleRegion(Region &region, Scop &scop, unsigned depth) {
  isl::schedule sched;
  for (Block &block : region.getBlocks())
    sched = combineInSequence(sched, scheduleBlock(block, scop, depth));
  return sched;
}

isl::schedule scheduleOperation(Operation *op, Scop &scop, unsigned depth) {
  isl::schedule sched;

  for (Region &region : op->getRegions()) {
    sched = combineInSequence(sched, scheduleRegion(region, scop, depth));
  }

  ScopStmt* stmt = scop.lookupStmt(op);
  if (stmt) { 
    if (seenStmts.find(stmt) == seenStmts.end()) {
      seenStmts[stmt] = true;
  
      isl::union_set domain;
      if (sched.is_null()) {
        domain = stmt->domain();
        sched = isl::schedule::from_domain(domain);
      } else {
        domain = sched.get_domain();
      }
  
      //isl::multi_union_pw_aff mupa = mapToDimension(domain, depth);
      //sched = sched.insert_partial_schedule(mupa);
    }

  } else if (isa<AffineForOp>(op)) {
    isl::multi_union_pw_aff mupa = mapToDimension(sched.get_domain(), depth);
    sched = sched.insert_partial_schedule(mupa);
  }

  return sched;
}

void ScopStmt::dump(raw_ostream &os, bool withLabels = true,
                    bool withName = true, bool withDomain = true,
                    bool withAccessOps = false) {
  if (withName)
    os << name << "\n";

  if (withDomain) {
    if (withLabels)
      os << "- Domain\n";
    os << "    " << stringFromIslObj(domain()) << "\n";
  }

  if (withAccessOps) {
    if (withLabels)
      os << "- Access Operations\n";
    for (unsigned i = 0; i < accessOps.size(); ++i) {
      os << "    ";
      accessOps[i]->print(os, OpPrintingFlags());
      os << "\n";
      os << "    " << stringFromIslObj(accessRelations[i]) << "\n";
    }
  }

  if (withLabels)
    os << "- Operations\n";
  for (unsigned i = 0; i < allOps.size(); ++i) {
    os << "    ";
    allOps[i]->print(os, OpPrintingFlags());
    os << "\n";
  }
}

Scop::Scop(Operation *op) : op(op) {
  ctx = isl_ctx_alloc();
  asmState = new AsmState(op);

  buildScopStmts();
  buildAccessRelationIslMaps();

  schedule = scheduleOperation(op, *this, 0);
  flowDependencies = computeFlowDependencies();
}

ScopStmt* Scop::lookupStmt(mlir::Operation *op) {
  for (auto &stmt : stmts) {
    for (auto &iop : stmt.accessOps) {
        if (op == iop) {
          return &stmt;
        }
    }
  }
  return nullptr;
}

llvm::SmallVector<ScopStmt> Scop::lookupStmts(mlir::Block &block) {
  llvm::SmallVector<Operation *> ops;
  for (Operation &iop : block) {
    ops.push_back(&iop);
  }

  llvm::DenseMap<ScopStmt *, bool> stmtsMap;
  for (auto &stmt : stmts) {
    for (auto &op : stmt.accessOps) {
      for (auto &iop : ops) {
        if (op == iop) {
          stmtsMap[&stmt] = true;
        }
      }
    }
  }
  llvm::SmallVector<ScopStmt> result;
  for (auto stmt : stmtsMap) {
    result.push_back(*stmt.first);
  }

  return result;
}

isl::union_map Scop::computeFlowDependencies() {
  isl::union_map reads;
  isl::union_map writes;
  for (auto &stmt : stmts) {
    for (unsigned i = 0; i < stmt.accessOps.size(); ++i) {
      Operation *op = stmt.accessOps[i];
      isl::union_map acc = stmt.accessRelations[i];
      if (isa<AffineLoadOp>(op)) {
        if (reads.is_null()) {
          reads = acc;
        } else {
          reads = reads.unite(acc);
        }
      } else if (isa<AffineStoreOp>(op)) {
        if (writes.is_null()) {
          writes = acc;
        } else {
          writes = writes.unite(acc);
        }
      }
    }
  }

  isl::union_access_info uai(reads);
  uai = uai.set_must_source(writes);
  uai = uai.set_schedule(schedule);
  isl::union_flow flow = uai.compute_flow();

  return flow.get_may_dependence();
}

void Scop::buildScopStmts() {
  llvm::SmallVector<Operation *> currentStmtAllOps;
  llvm::SmallVector<Operation *> currentStmtAccessOps;

  unsigned stmtIdx = 0;
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool isSimpleOp = !isa<AffineForOp>(op) && !isa<AffineYieldOp>(op) &&
        !isa<FunctionOpInterface>(op) && !isa<ModuleOp>(op) &&
        !isa<func::ReturnOp>(op);
    bool isAccessOp = isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op);
    bool isStoreOp = isa<AffineStoreOp>(op);

    if (isSimpleOp) {
      currentStmtAllOps.push_back(op);
    }

    if (isSimpleOp && isAccessOp) {
      currentStmtAccessOps.push_back(op);
    }

    if ((!isSimpleOp || isStoreOp)) {
      if (!currentStmtAccessOps.empty()) {
        ScopStmt scopStmt(currentStmtAllOps, currentStmtAccessOps,
                          "Stmt" + std::to_string(stmtIdx++));
        stmts.push_back(scopStmt);
      }
      currentStmtAccessOps.clear();
      currentStmtAllOps.clear();
    }
  });

  if (!currentStmtAccessOps.empty()) {
    ScopStmt scopStmt(currentStmtAllOps, currentStmtAccessOps,
                      "Stmt" + std::to_string(stmtIdx++));
    stmts.push_back(scopStmt);
  }
}

isl::map Scop::getAccessRelation(Operation *op, std::string &opName) {
  // Get the access relation using Presburger lib.
  MemRefAccess access(op);
  FlatAffineRelation rel;
  LogicalResult result = access.getAccessRelation(rel);
  assert(succeeded(result) && "Failed to get access relation");
  // dumpRelDetails(rel);

  // Create access relation space.
  isl::space accessRelSpace =
      isl::space(ctx, rel.getNumSymbolVars(), rel.getNumDomainDims(),
                 rel.getNumRangeDims());

  // Set tuple names.
  isl::id opId = isl::id::alloc(ctx, opName, op);
  isl::id memrefId;
  // std::string memrefName =
  //     "MemRef" +
  //     std::to_string(access.memref.cast<BlockArgument>().getArgNumber());
  std::string memrefName = "MemRef";
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    memrefId = isl::id::alloc(ctx, memrefName, loadOp.getMemRef().getImpl());
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    memrefId = isl::id::alloc(ctx, memrefName, storeOp.getMemref().getImpl());
  } else {
    llvm_unreachable("unexpected operation");
  }
  accessRelSpace =
      accessRelSpace.set_tuple_id(isl::dim::in, isl::id(ctx, opName));

  // Set input identifiers.
  for (unsigned i = 0; i < rel.getNumDomainDims(); ++i) {
    std::string asmId;
    raw_string_ostream so(asmId);
    rel.getValue(i).printAsOperand(so, *asmState);
    asmId = std::regex_replace(asmId, std::regex(R"(%)"), "");

    std::string id = asmId;
    // std::string id = "i" + std::to_string(i);

    isl::id islId =
        isl::id::alloc(ctx, id, rel.getValue(i).getImpl());
    accessRelSpace = accessRelSpace.set_dim_id(isl::dim::in, i, islId);
  }

  // Set param identifiers.
  for (unsigned i = 0; i < rel.getNumSymbolVars(); ++i) {
    // isl::id id =
    //     isl::id::alloc(ctx, "p_" + std::to_string(i),
    //                    rel.getValue(rel.getNumDimVars() + i).getImpl());
    accessRelSpace = accessRelSpace.set_dim_id(isl::dim::param, i, isl::id(ctx, "p_" + std::to_string(i)));
  }

  // Convert the access relation to isl::map.
  isl::basic_map accessRel = isl::basic_map::universe(accessRelSpace);
  unsigned localOffsetISL = rel.getNumRangeDims();
  accessRel = accessRel.insert_dims(isl::dim::out, localOffsetISL,
                                    rel.getNumLocalVars());

  // Copy inequality constraints.
  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i) {
    isl::constraint c =
        isl::constraint::alloc_inequality(accessRel.get_local_space());
    unsigned offset = 0;
    // Domain variables.
    for (unsigned j = 0, e = rel.getNumDomainDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::in, j, rel.atIneq64(i, offset + j));
    offset += rel.getNumDomainDims();
    // Range variables.
    for (unsigned j = 0, e = rel.getNumRangeDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j, rel.atIneq64(i, offset + j));
    offset += rel.getNumRangeDims();
    // Symbol variables.
    for (unsigned j = 0, e = rel.getNumSymbolVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::param, j, rel.atIneq64(i, offset + j));
    offset += rel.getNumSymbolVars();
    // Local variables.
    for (unsigned j = 0, e = rel.getNumLocalVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j + localOffsetISL,
                               rel.atIneq64(i, offset + j));

    c = c.set_constant_si(rel.atIneq64(i, rel.getNumCols() - 1));
    accessRel = accessRel.add_constraint(c);
  }

  // Copy equality constraints.
  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i) {
    isl::constraint c =
        isl::constraint::alloc_equality(accessRel.get_local_space());
    unsigned offset = 0;
    // Domain variables.
    for (unsigned j = 0, e = rel.getNumDomainDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::in, j, rel.atEq64(i, offset + j));
    offset += rel.getNumDomainDims();
    // Range variables.
    for (unsigned j = 0, e = rel.getNumRangeDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j, rel.atEq64(i, offset + j));
    offset += rel.getNumRangeDims();
    // Symbol variables.
    for (unsigned j = 0, e = rel.getNumSymbolVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::param, j, rel.atEq64(i, offset + j));
    offset += rel.getNumSymbolVars();
    // Local variables.
    for (unsigned j = 0, e = rel.getNumLocalVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j + localOffsetISL,
                               rel.atEq64(i, offset + j));

    c = c.set_constant_si(rel.atEq64(i, rel.getNumCols() - 1));
    accessRel = accessRel.add_constraint(c);
  }

  isl::map accessRelM = isl::map(accessRel);
  accessRelM = accessRelM.project_out(isl::dim::out, localOffsetISL,
                                      rel.getNumLocalVars());
  accessRelM = accessRelM.set_tuple_id(isl::dim::out, memrefId);

  return accessRelM;
}

void Scop::buildAccessRelationIslMaps() {
  for (auto &stmt : stmts) {
    llvm::SmallVector<isl::map> ms;
    for (auto &op : stmt.accessOps) {
      // Build ISL access relation.
      isl::map m = getAccessRelation(op, stmt.name);
      ms.push_back(m);
    }
    stmt.accessRelations = ms;
  }
}

void Scop::dump(raw_ostream &os) {
  for (auto stmt : stmts) {
    stmt.dump(os);
    os << "\n";
  }

  os << "Schedule\n";
  os << "    " << stringFromIslObj(schedule) << "\n\n";
  os << "    " << stringFromIslObj(schedule.get_map()) << "\n\n";
  dumpIslObj(schedule.get_root(), os);

  os << "FlowDependences\n";
  for (auto m : flowDependencies.get_map_list()) {
    os << "    " << stringFromIslObj(m) << "\n";
  }

  os << "\n";
}

void Scop::dumpRelDetails(FlatAffineRelation rel) {
  llvm::outs() << "--------Relation--------\n";

  rel.dump();
  for (unsigned int i = 0; i < rel.getNumVars(); ++i) {
    llvm::outs() << "Var " << i << "\t";

    std::string varKind;
    switch (rel.getVarKindAt(i)) {
    case mlir::presburger::VarKind::Symbol:
      varKind = "Symbol";
      break;
    case mlir::presburger::VarKind::Local:
      varKind = "Local";
      break;
    case mlir::presburger::VarKind::Domain:
      varKind = "Domain";
      break;
    case mlir::presburger::VarKind::Range:
      varKind = "Range";
      break;
    }
    llvm::outs() << varKind << "\t";

    if (rel.hasValue(i)) {
      rel.getValue(i).dump();
    } else {
      llvm::outs() << "\n";
    }
  }

  llvm::outs() << "Dims: " << rel.getNumDomainDims() << " "
               << rel.getNumRangeDims() << "\n";
}
