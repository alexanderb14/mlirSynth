set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# Lib: ArgTuples
add_library(CE enumeration/ArgTuples.cc)
target_include_directories(CE PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Lib: Polyhedral Analysis
set(POLY_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${test_libs}
    MLIRAffineAnalysis
    MLIRAnalysis
    MLIRDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR)
add_library(polyhedral-analysis analysis/PolyhedralAnalysis.cc
                                analysis/isl/isl_helper.cc)
target_include_directories(polyhedral-analysis
                           PUBLIC ${CMAKE_SOURCE_DIR}/isl-0.25/include)
target_link_libraries(
  polyhedral-analysis PUBLIC ${POLY_LIBS}
                             ${CMAKE_SOURCE_DIR}/isl-0.25/.libs/libisl.a gmp)
llvm_update_compile_flags(polyhedral-analysis)
target_compile_options(polyhedral-analysis
                       PRIVATE -Wno-missing-field-initializers)

# Lib: Transform
add_library(
  transform
  transforms/CopyModifiedMemrefsPass.cc transforms/MemrefMinifyPass.cc
  transforms/LoopDistributionPass.cc transforms/LoopOutlinePass.cc Utils.cc)
target_include_directories(transform PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(transform PUBLIC polyhedral-analysis)
llvm_update_compile_flags(transform)

# Exe: Transform Tool
add_llvm_executable(opt tools/Opt.cc)
target_include_directories(opt PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                                       ${Boost_INCLUDE_DIR})
target_link_libraries(
  opt
  PRIVATE ${SYNTH_LIBS}
          CE
          ${MHLO_EXPORTED_TARGETS}
          ${POLY_LIBS}
          polyhedral-analysis
          transform
          ${Boost_LIBRARIES})

# Exe: Synthesizer
set(SYNTH_LIBS ${dialect_libs} ${conversion_libs} MLIROptLib)
add_llvm_executable(
  synthesizer
  tools/Synthesizer.cc
  ContextManager.cc
  enumeration/AttributeGen.cc
  enumeration/Candidate.cc
  enumeration/Enumerator.cc
  enumeration/Guide.cc
  enumeration/Stats.cc
  execution/ArgUtils.cc
  execution/ArrayUtils.cc
  execution/Executor.cc
  execution/Lowering.cc)
target_include_directories(synthesizer PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                                               ${Boost_INCLUDE_DIR})
target_link_libraries(
  synthesizer
  PRIVATE ${SYNTH_LIBS}
          CE
          ${MHLO_EXPORTED_TARGETS}
          ${POLY_LIBS}
          polyhedral-analysis
          transform
          ${Boost_LIBRARIES})
llvm_update_compile_flags(synthesizer)
