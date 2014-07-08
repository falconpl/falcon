####################################################################
# The Falcon Programming Language
#
# CMake configuration file for gd2
####################################################################

cmake_minimum_required(VERSION 2.6)

if(NOT FALCON_BUILD_GD2 )
	message( STATUS "NOT building module: gd2")
	return()
endif()
message( STATUS "Building module: gd2" )

PROJECT(gd2)

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

#process source directory
ADD_SUBDIRECTORY(src)

