####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

cmake_minimum_required(VERSION 3.16.3)
project(Falcon_dynlib)

find_package(Falcon REQUIRED)

# Set here project-wide options
option(FALCON_DYNLIB_WITH_TESTS  "Compile test_dynlib.c to generate the test library" OFF)
if(FALCON_WITH_CTEST_TESTS)
  set(FALCON_DYNLIB_WITH_TESTS on)
endif()

#process source directory
add_subdirectory(src)

if(FALCON_DYNLIB_WITH_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()
