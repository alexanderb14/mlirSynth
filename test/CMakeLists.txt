configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(IRSYNTH_TEST_DEPENDS FileCheck opt synthesizer)

add_lit_testsuite(check-transforms "Running transforms tests"
                  ${CMAKE_CURRENT_BINARY_DIR}/transforms DEPENDS ${IRSYNTH_TEST_DEPENDS})

add_lit_testsuite(check-synthesis "Running synthesis tests"
                  ${CMAKE_CURRENT_BINARY_DIR}/synthesis DEPENDS ${IRSYNTH_TEST_DEPENDS})

add_lit_testsuite(check-equiv "Running equivalence tests"
                  ${CMAKE_CURRENT_BINARY_DIR}/equiv DEPENDS ${IRSYNTH_TEST_DEPENDS})