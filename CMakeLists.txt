####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################
cmake_minimum_required(VERSION 2.6.2)
PROJECT(Falcon_Feathers)

# Find our Find*.cmake files 
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

find_package(Falcon
   HINTS "$ENV{FALCON_ACTIVE_TREE}/*/*/cmake"
         "$ENV{FALCON_ACTIVE_TREE}/*/cmake"
         "$ENV{FALCON_ACTIVE_TREE}/cmake"
   REQUIRED)

add_subdirectory(compiler)
add_subdirectory(confparser)
add_subdirectory(funcext)
add_subdirectory(json)
add_subdirectory(logging)
add_subdirectory(mxml)
add_subdirectory(process)
add_subdirectory(regex)
add_subdirectory(socket)
add_subdirectory(threading)
add_subdirectory(zlib)

# install headers
add_subdirectory(include)

option(FALCON_WITH_CTEST_TESTS "" off)
if(FALCON_WITH_CTEST_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

 
