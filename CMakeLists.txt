####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dynlib
####################################################################

cmake_minimum_required(VERSION 2.6)
project(Falcon_dynlib)

find_package(falcon REQUIRED)

# Set here project-wide options
option(WITH_TESTS  "Compile test_dynlib.c to generate the test library" OFF)
option(FALCON_WITH_CTEST_TESTS "" off)
if(FALCON_WITH_CTEST_TESTS)
  set(WITH_TESTS on)
endif()

#process source directory
add_subdirectory(src)

if(WITH_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif(WITH_TESTS)
