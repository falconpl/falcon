####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

if(NOT FALCON_MODULE_CURL)
	message( STATUS "NOT building module: dtmx")
	return()
end()

cmake_minimum_required(VERSION 2.6)
project(Falcon_curl)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Falcon REQUIRED)
find_package(CURL 7.18.2 REQUIRED)

## process source directories
# add other subdirectories that contain libraries on which the final module depends here..
add_subdirectory(src)


