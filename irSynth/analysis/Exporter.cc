#include "Exporter.h"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <regex>
#include <string>

using namespace llvm;
using namespace mlir;

void toDot(raw_ostream &os, Scop &scop) {
  os << "digraph {\n";

  // Edges.
  for (auto um : scop.flowDependencies.get_map_list()) {
    isl::map m = um.as_map();

    os << "  " << stringFromIslObj(m.get_tuple_id(isl::dim::in)) << " -> "
       << stringFromIslObj(m.get_tuple_id(isl::dim::out));

    // m = m.set_tuple_id(isl::dim::in, isl::id(ctx, "s"));
    // m = m.set_tuple_id(isl::dim::out, isl::id(ctx, "s"));

    std::string label = "";
    label += stringFromIslObj(m) + "\\l";

    os << " [label=\"" << label << "\"];\n";
  }

  // Nodes.
  for (auto &stmt : scop.stmts) {
    os << "  " << stmt.name;
    os << " [shape=box, ";

    os << "label=\"";
    std::string stmtStr;
    raw_string_ostream rso(stmtStr);
    stmt.dump(rso, false, false, false, false);

    // Remove type annotations
    stmtStr = std::regex_replace(stmtStr, std::regex(R"( :.*\n)"), "\n");
    // Remove leading whitespaces before instructions
    stmtStr = std::regex_replace(stmtStr, std::regex(R"(\n[\s]*)"), "\n");
    stmtStr = std::regex_replace(stmtStr, std::regex(R"(^[\s]*)"), "");

    // Align left in dot graph
    stmtStr = std::regex_replace(stmtStr, std::regex(R"(\n)"), "\\l");
    os << stmtStr;
    os << "\"];\n";
  }

  os << "}\n";
}

void toDotStmts(raw_ostream &os, Scop &scop) {
  os << "digraph {\n";

  mlir::DenseMap<mlir::Operation *, std::string> opStrs;
  for (auto &stmt : scop.stmts) {
    os << "subgraph cluster_" + stmt.name << " {\n";
    os << "  label = \"" << stmt.name << "\";\n";

    // Nodes.
    for (auto &op : stmt.allOps) {
      if (opStrs.find(op) == opStrs.end()) {
        std::string opStr = "n" + std::to_string(opStrs.size());
        opStrs[op] = opStr;

        os << "  " << opStr;
        os << " [shape=box, ";
        os << "label=\"" << op->getName().getStringRef().str() << "\"";
        os << "];\n";
      }
    }

    // Edges.
    for (auto &op : stmt.allOps) {
      // Definitions from ops
      for (auto operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (opStrs.find(defOp) != opStrs.end()) {
            os << "  " << opStrs[defOp] << " -> " << opStrs[op] << ";\n";
          }
        }
      }
    }

    os << "}\n";
  }

  // External vars or args
  mlir::DenseMap<mlir::Value, std::string> undefArgs;
  mlir::DenseMap<mlir::Operation *, std::string> undefOps;
  os << "subgraph cluster_external {\n";
  os << "  label = \"External Vars and Args\";\n";
  for (auto &stmt : scop.stmts) {
    for (auto &op : stmt.allOps) {
      for (auto operand : op->getOperands()) {
        mlir::Operation *iop = operand.getDefiningOp();
        // It's a variable
        if (iop) {
          if (opStrs.find(iop) == opStrs.end()) {
            // Create node
            if (undefOps.find(iop) == undefOps.end()) {
              std::string undefStr =
                  "var" + std::to_string(undefArgs.size() + undefOps.size());
              undefOps[iop] = undefStr;

              os << "  " << undefStr;
              os << " [shape=box, ";
              os << "label=\"Ext\"";
              os << "];\n";
            }
            // Create edge
            os << "  " << undefOps[iop] << " -> " << opStrs[op] << ";\n";
          }
          // It's an argument
        } else {
          auto blockArg = operand.cast<BlockArgument>();

          // Create node
          if (undefArgs.find(blockArg) == undefArgs.end()) {
            std::string undefStr =
                "var" + std::to_string(undefArgs.size() + undefOps.size());
            undefArgs[blockArg] = undefStr;

            os << "  " << undefStr;
            os << " [shape=box, ";
            os << "label=\"Ext\"";
            os << "];\n";
          }
          // Create edge
          os << "  " << undefArgs[blockArg] << " -> " << opStrs[op] << ";\n";
        }
      }
    }
  }
  os << "}\n";

  os << "}\n";
}
