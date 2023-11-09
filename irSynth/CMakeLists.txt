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

# Lib: Synthesis
add_library(cartesian_product
  synthesis/CartesianProduct.cc)
target_include_directories(cartesian_product
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})

add_library(synthesis
  synthesis/Generators.cc
  synthesis/Candidate.cc
  synthesis/CandidateStore.cc
  synthesis/Synthesizer.cc
  synthesis/Guide.cc
  synthesis/Grammar.cc
  synthesis/ProcessingStatus.cc
  synthesis/Spec.cc
  synthesis/Stats.cc)
target_include_directories(synthesis
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}
          ${CMAKE_SOURCE_DIR}/indicators/single_include
          ${MHLO_INCLUDE_DIRS})
target_link_libraries(synthesis
  PRIVATE cartesian_product
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
  transforms/AnnotateLastStoredMemrefArgPass.cc
  transforms/ChangeSizesPass.cc
  transforms/CleanupPass.cc
  transforms/CopyModifiedMemrefsPass.cc
  transforms/FoldToTensorToMemrefPairPass.cc
  transforms/LoopDistributionPass.cc
  transforms/LoopOutlinePass.cc
  transforms/MemrefCopyToLoopsPass.cc
  transforms/PrepareTargetPass.cc
  transforms/TargetOutlinePass.cc
  transforms/Utils.cc)
target_include_directories(transform
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(transform
  PRIVATE analysis)
llvm_update_compile_flags(transform)

# Lib: Validation
add_library(validation
  validation/CheckingValidator.cc
  validation/TestingValidator.cc)
target_include_directories(validation
  PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(validation
  PRIVATE MLIRTargetCpp)

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
          synthesis
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
          synthesis
          execution
          transform
          validation)
llvm_update_compile_flags(test-executor)

# Exe: Grammar Extractor
set(LLVM_LINK_COMPONENTS TableGen)
add_llvm_executable(grammar-extractor
  tools/GrammarExtractor.cc)
target_link_libraries(grammar-extractor
  PRIVATE MLIRIR
          MLIRTableGen
          MLIRTblgenLib
          MLIRSupport)
