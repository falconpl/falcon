####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib - TESTS
####################################################################
cmake_minimum_required(VERSION 3.16.3)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${falcon_INCLUDE_DIRS}
)

add_library( test_dynlib MODULE
   testsrc/test_dynlib.c
)
# dynlib goes in the testsuite source.# this is because we need it for our 
# tests.
set_target_properties(test_dynlib PROPERTIES 
  PREFIX ""
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/testsuite
)
      
if(APPLE)
  set_target_properties(test_dynlib PROPERTIES
    SUFFIX ".dylib" )
endif()

add_subdirectory(testsuite)
