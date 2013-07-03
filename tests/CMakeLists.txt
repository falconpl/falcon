add_subdirectory(core/testsuite)
add_subdirectory(core/longtests)
add_subdirectory(core/samples)

#ok testsuite should run, but should we install it?
if( FALCON_INSTALL_TESTS )
   install(DIRECTORY . DESTINATION ${Falcon_SHARE_DIR}/tests )
endif()