
###########################  NOTICE  ################################## 
# 
# This CMakeLists.txt refers to falcon script tests only.
#
# The bin/ directory is under the tests/ dir because of the logic
# order of the source tree, but it's directly controlled by the 
# toplevel  CMakeLists.txt entry
#     

add_subdirectory(testsuite)
add_subdirectory(longtests)
add_subdirectory(samples)

#ok testsuite should run, but should we install it?
if( FALCON_INSTALL_TESTS )
   install(DIRECTORY . DESTINATION ${Falcon_SHARE_DIR}/tests )
endif()

