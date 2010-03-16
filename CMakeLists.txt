####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

cmake_minimum_required(VERSION 2.6)
project(falcon-module-hpdf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(Falcon REQUIRED)
find_package(LibHaru REQUIRED)

add_subdirectory(src)

