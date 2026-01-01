####################################################################
# The Falcon Programming Language
#
# CMake configuration file for dbus
####################################################################

PROJECT(dbus)
cmake_minimum_required(VERSION 3.16.3)

INCLUDE( ${CMAKE_ROOT}/Modules/FindPkgConfig.cmake)

# Set here project-wide options
# OPTION( WITH_OPT  "An poption" OFF)

#INCLUDE( "falmod.cmake" )

# Search for dbus
pkg_check_modules(DBUS REQUIRED dbus-1 )
MESSAGE( "Found DBUS: ${DBUS_INCLUDE_DIRS}" )

#process source directory
ADD_SUBDIRECTORY(src)

