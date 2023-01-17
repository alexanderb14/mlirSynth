set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_library(CE enumeration/ArgTuples.cc)
target_include_directories(CE PRIVATE ${CMAKE_CURRENT_LIST_DIR})

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
  enumeration/Stats.cc
  execution/ArgUtils.cc
  execution/ArrayUtils.cc
  execution/Executor.cc
  execution/Lowering.cc
  Utils.cc)
target_include_directories(synthesizer PRIVATE ${CMAKE_CURRENT_LIST_DIR})
llvm_update_compile_flags(synthesizer)
target_link_libraries(synthesizer PRIVATE ${SYNTH_LIBS} CE ${MHLO_EXPORTED_TARGETS})

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
add_llvm_executable(affine-info
  analysis/affine-info.cpp
  analysis/Scop.cpp
  analysis/Exporter.cpp
  analysis/isl/isl_helper.cpp
  )
target_include_directories(affine-info PRIVATE /devel/tmp/isl/include)
llvm_update_compile_flags(affine-info)
target_link_libraries(affine-info PRIVATE ${POLY_LIBS} /devel/tmp/isl/.libs/libisl.a gmp)
target_compile_options(affine-info PRIVATE -Wno-missing-field-initializers)
