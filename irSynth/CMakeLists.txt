set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(SYNTH_LIBS ${dialect_libs} ${conversion_libs} MLIROptLib)

# Lib: Analysis
add_library(analysis
  analysis/PolyhedralAnalysis.cc
  analysis/isl/isl_helper.cc)
target_include_directories(analysis
  PUBLIC ${CMAKE_SOURCE_DIR}/isl-0.25/include)
target_link_libraries(analysis
  PRIVATE ${dialect_libs}
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
          ${CMAKE_SOURCE_DIR}/isl-0.25/.libs/libisl.a
          gmp)
llvm_update_compile_flags(analysis)
target_compile_options(analysis
  PRIVATE -Wno-missing-field-initializers)

# Lib: Enumeration
add_library(argtuples
  enumeration/ArgTuples.cc)
target_include_directories(argtuples
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_library(enumeration
  enumeration/Generators.cc
  enumeration/Candidate.cc
  enumeration/Enumerator.cc
  enumeration/Guide.cc
  enumeration/Stats.cc
  enumeration/Utils.cc)
target_include_directories(enumeration
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(enumeration
  PRIVATE argtuples
          analysis)

# Lib: Execution
add_library(execution
  execution/ArgUtils.cc
  execution/ArrayUtils.cc
  execution/Executor.cc
  execution/Lowering.cc)
target_include_directories(execution
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

# Lib: Transform
add_library(transform
  transforms/CopyModifiedMemrefsPass.cc
  transforms/MemrefMinifyPass.cc
  transforms/LoopDistributionPass.cc
  transforms/LoopOutlinePass.cc
  transforms/CleanupPass.cc
  transforms/Utils.cc)
target_include_directories(transform
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(transform
  PRIVATE analysis)
llvm_update_compile_flags(transform)

# Exe: Transform Tool
add_llvm_executable(opt
  tools/Opt.cc)
target_include_directories(opt
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(opt
  PRIVATE ${SYNTH_LIBS}
          ${MHLO_EXPORTED_TARGETS}
          analysis
          transform)

# Exe: Synthesizer
add_llvm_executable(synthesizer
  tools/Synthesizer.cc
  ContextManager.cc)
target_include_directories(synthesizer
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(synthesizer
  PRIVATE ${SYNTH_LIBS}
          ${MHLO_EXPORTED_TARGETS}
          analysis
          enumeration
          execution
          transform)
llvm_update_compile_flags(synthesizer)

# Exe: Test Executor
add_llvm_executable(test-executor
  tools/TestExecutor.cc
  ContextManager.cc)
target_include_directories(test-executor
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(test-executor
  PRIVATE ${SYNTH_LIBS}
          ${MHLO_EXPORTED_TARGETS}
          analysis
          enumeration
          execution
          transform)
llvm_update_compile_flags(test-executor)

# Exe: Tablegen Extractor
set(LLVM_LINK_COMPONENTS TableGen)
add_llvm_executable(tablegen-extractor
  tools/TablegenExtractor.cc)
target_link_libraries(tablegen-extractor
  PRIVATE MLIRIR
          MLIRTableGen
          MLIRTblgenLib
          MLIRSupport)
