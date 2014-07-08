####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dbus
####################################################################

cmake_minimum_required(VERSION 2.6)

FalconDeclareModule( DBUS dbus )

PROJECT(dbus)
INCLUDE( ${CMAKE_ROOT}/Modules/FindPkgConfig.cmake)

# Set here project-wide options
# OPTION( WITH_OPT  "An poption" OFF)

# Search for dbus
pkg_check_modules(DBUS REQUIRED dbus-1 )
MESSAGE( "Found DBUS: ${DBUS_INCLUDE_DIRS}" )

#process source directory
ADD_SUBDIRECTORY(src)

