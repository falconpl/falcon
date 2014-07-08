####################################################################
# The Falcon Programming language
#
# CMake configuration file for Feather modules
####################################################################

cmake_minimum_required(VERSION 2.6)

if(NOT FALCON_BUILD_HPDF )
	message( STATUS "NOT building module: hpdf")
	return()
endif()
message( STATUS "Building module: hpdf" )

project(falcon-module-hpdf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(LibHaru REQUIRED)

add_subdirectory(src)

