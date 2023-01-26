configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SYNTHESIZER_TEST_DEPENDS
        FileCheck count not
        synthesizer
        )

add_lit_testsuite(check-synthesizer "Running the synthesizer tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${SYNTHESIZER_TEST_DEPENDS}
        )
set_target_properties(check-synthesizer PROPERTIES FOLDER "Tests")

add_lit_testsuites(SYNTHESIZER ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${SYNTHESIZER_TEST_DEPENDS})
