get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LLVM_LINK_COMPONENTS
  Core
  Support
  AsmParser
  )

set(LIBS
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
  affine-info.cpp
  Scop.cpp
  Exporter.cpp
  isl/isl_helper.cpp

  DEPENDS
  ${LIBS}
  )
include_directories(affine-info PRIVATE /devel/tmp/isl/include)
target_link_libraries(affine-info PRIVATE ${LIBS})
target_link_libraries(affine-info PRIVATE /devel/tmp/isl/.libs/libisl.a gmp)
target_compile_options(affine-info PRIVATE -Wno-missing-field-initializers)
llvm_update_compile_flags(affine-info)
