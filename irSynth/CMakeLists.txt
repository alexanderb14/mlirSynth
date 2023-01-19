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
  MLIRIR
  )
add_library(polyhedral-analysis
  analysis/PolyhedralAnalysis.cc
  analysis/isl/isl_helper.cc)
target_include_directories(polyhedral-analysis PUBLIC ${CMAKE_SOURCE_DIR}/isl-0.25/include)
llvm_update_compile_flags(polyhedral-analysis)
target_link_libraries(polyhedral-analysis PUBLIC ${POLY_LIBS}
  ${CMAKE_SOURCE_DIR}/isl-0.25/.libs/libisl.a gmp)
target_compile_options(polyhedral-analysis PRIVATE -Wno-missing-field-initializers)

# Exe: Polyhedral Analysis Tool
add_llvm_executable(polyhedral-analysis-tool
  analysis/PolyhedralInfo.cc)
target_link_libraries(polyhedral-analysis-tool PRIVATE polyhedral-analysis)

# Exe: Synthesizer
set(SYNTH_LIBS ${dialect_libs}
  ${conversion_libs}
  MLIROptLib)
add_llvm_executable(
  synthesizer
  main.cc
  ContextManager.cc
  enumeration/AttributeGen.cc
  enumeration/Candidate.cc
  enumeration/Enumerator.cc
  enumeration/Guide.cc
  enumeration/Stats.cc
  execution/ArgUtils.cc
  execution/ArrayUtils.cc
  execution/Executor.cc
  execution/Lowering.cc
  transforms/LoopDistributionPass.cc
  transforms/LoopOutlinePass.cc
  Utils.cc)
target_include_directories(synthesizer PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${Boost_INCLUDE_DIR})
llvm_update_compile_flags(synthesizer)
target_link_libraries(synthesizer PRIVATE ${SYNTH_LIBS} CE ${MHLO_EXPORTED_TARGETS} ${POLY_LIBS} polyhedral-analysis ${Boost_LIBRARIES})
