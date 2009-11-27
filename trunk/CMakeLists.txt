####################################################################
# FALCON - The Falcon Programming Language.
#
# CMake configuration file for curl
####################################################################

PROJECT(curl)
cmake_minimum_required(VERSION 2.6)

find_package( CURL
   PATHS 
      ./libcurl
      )

if(CURL_FOUND)
   Message( "Found curl in ${CURL_INCLUDE_DIRS}" )
else()
   Message( "Please, specify CURL_INCLUDE_DIRS and CURL_LIBRARY variables" )
   Message( FATAL_ERROR "Can't find CURL library" )
endif()

# Set here project-wide options
# OPTION( WITH_OPT  "An poption" OFF)

INCLUDE( "falmod.cmake" )

#process source directory
ADD_SUBDIRECTORY(src)

#add other subdirectories here.
