import argparse

header = """
#include "ArgTuples.h"

#include "enumeration/Generators.h"
#include "enumeration/Utils.h"

#include <range/v3/all.hpp>
#include <range/v3/view/cartesian_product.hpp>

#include <iostream>
#include <vector>

using namespace llvm;
using namespace mlir;

"""

body_header = """
std::vector<ArgTuple>
getOperandArgTuples(MLIRContext &ctx, RegisteredOperationName opName,
                          std::vector<CandidatePtr> &operandCandidates) {
  OpBuilder builder(&ctx);
  Operation *op =
      builder.create(UnknownLoc::get(&ctx), opName.getIdentifier(), {});

  int numOperands = getRequiredNumOperands(op);
  int numAttributes = getRequiredNumAttributes(op);

  Block::BlockArgListType blockArgs;
  std::vector<Attribute> attributeCandidates =
      genAttributes(builder, blockArgs, 2);
  int numRegions = getRequiredNumRegions(op);

  std::vector<std::shared_ptr<Region>> regionCandidates = genRegions(builder);
"""

body_footer = """
  llvm::outs() << "Unsupported number of operands (" << numOperands
               << "), attributes (" << numAttributes << "), regions ("
               << numRegions << ") in op: " << opName.getIdentifier() << "\\n";
  assert(false);
}
"""
 

def get_function(numOperands, numAttributes, numRegions):
    src = ""
    src += "std::vector<ArgTuple>\n"
    src += "get{}operands{}attributes{}regions(std::vector<CandidatePtr> &operandCandidates,\n".format(numOperands, numAttributes, numRegions)
    src += "                       std::vector<Attribute> &attributeCandidates,\n"
    src += "                       std::vector<std::shared_ptr<Region>> &regionCandidates) {\n"
    src += "  auto cands =\n"
    src += "      ranges::views::cartesian_product("
    for i in range(numOperands):
        src += "operandCandidates, "
    for i in range(numAttributes):
        src += "attributeCandidates, "
    for i in range(numRegions):
        src += "regionCandidates, "
    src = src[:-2]
    src += ");\n"
    src += "  std::vector<ArgTuple> ret;\n"
    src += "  for (auto cand : cands) {\n"
    src += "    ArgTuple tuple;\n"

    counter = 0
    if numOperands > 0:
        src += "    tuple.operands = {"
        for i in range(numOperands):
            src += "std::get<{}>(cand), ".format(counter)
            counter += 1
        src = src[:-2]
        src += "};\n"
    if numAttributes > 0:
        src += "    tuple.attributes = {"
        for i in range(numAttributes):
            src += "std::get<{}>(cand), ".format(counter)
            counter += 1
        src = src[:-2]
        src += "};\n"
    if numRegions > 0:
        src += "    tuple.regions = {"
        for i in range(numRegions):
            src += "std::get<{}>(cand), ".format(counter)
            counter += 1
        src = src[:-2]
        src += "};\n"

    src += "    ret.push_back(tuple);\n"
    src += "  }\n"
    src += "  return ret;\n"
    src += "}\n\n"
    return src


def get_call(numOperands, numAttributes, numRegions):
    src = ""
    src += "  if (numOperands == {} && numAttributes == {} && numRegions == {}) {{\n".format(numOperands, numAttributes, numRegions)
    src += "    return get{}operands{}attributes{}regions(operandCandidates, attributeCandidates, regionCandidates);\n".format(numOperands, numAttributes, numRegions)
    src += "  }\n"
    return src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_operands", type=int)
    parser.add_argument("--max_attributes", type=int)
    parser.add_argument("--max_regions", type=int)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    src = "// WARNING: DO NOT EDIT THIS FILE. IT IS AUTOGENERATED BY\n"
    src += "// scripts/gen_ArgTuples.py\n"
    src += header

    for i in range(0, args.max_operands):
        for j in range(0, args.max_attributes):
            for k in range(0, args.max_regions):
                if i == 0 and j == 0 and k == 0:
                    continue
                src += get_function(i, j, k)
    
    src += body_header
    for i in range(0, args.max_operands):
        for j in range(0, args.max_attributes):
            for k in range(0, args.max_regions):
                if i == 0 and j == 0 and k == 0:
                    continue
                src += get_call(i, j, k)
    src += body_footer

    with open(args.output, "w") as f:
        f.write(src)

if __name__ == "__main__":
    main()
