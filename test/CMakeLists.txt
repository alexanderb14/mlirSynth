configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(IRSYNTH_TEST_DEPENDS FileCheck count not opt synthesizer)
add_lit_testsuite(check-irsynth "Running irsynth tests"
                  ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${IRSYNTH_TEST_DEPENDS})
set_target_properties(check-irsynth PROPERTIES FOLDER "Tests")
add_lit_testsuites(IRSYNTH ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS
                   ${IRSYNTH_TEST_DEPENDS})
