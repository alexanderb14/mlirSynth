set(LLVM_LINK_COMPONENTS Core Support nativecodegen native)

add_library(CE enumeration/CandidateTuples.cc)
target_include_directories(CE PRIVATE ${CMAKE_CURRENT_LIST_DIR})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS ${dialect_libs} ${conversion_libs} MLIROptLib)

add_llvm_executable(
  synthesizer
  main.cc
  ContextManager.cc
  enumeration/Candidate.cc
  enumeration/Enumerator.cc
  enumeration/Stats.cc
  execution/ArgUtils.cc
  execution/Executor.cc
  execution/Lowering.cc
  Utils.cc)
target_include_directories(synthesizer PRIVATE ${CMAKE_CURRENT_LIST_DIR})
llvm_update_compile_flags(synthesizer)
target_link_libraries(synthesizer PRIVATE ${LIBS} CE ${MHLO_EXPORTED_TARGETS})

mlir_check_all_link_libraries(synthesizer)
