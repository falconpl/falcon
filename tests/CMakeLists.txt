add_subdirectory(core/testsuite)

#ok testsuite should run, but should we install it?
if( FALCON_INSTALL_TESTS )
   install(DIRECTORY . DESTINATION ${Falcon_SHARE_DIR}/tests )