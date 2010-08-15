#############################################################################
# The Falcon Programming language
#
# CMake configuration file for Core falcon
##############################################################################
#
##  Control Variables
#
# FALCON_BIN_DIR    - Prefix for binary installation. Defaults to "bin"
# FALCON_INC_DIR    - Prefix for installation of include files.
# FALCON_LIB_DIR    - Prefix for installation of archives and dynamic libraries
#                     on UNIXes. Defaults to "lib".
# FALCON_MOD_DIR    - Prefix for installation of modules. Defaults to
#                     $FALCON_LIB_DIR/falcon on UNICES, and bin/ on windows.
# FALCON_MAN_DIR    - Where manual pages are stored (below prefix)
# FALCON_SHARE_DIR  - Where to install falco share files
# FALCON_CMAKE_DIR  - Where to install FalconConfig.cmake & c
#
##  Options
#
# FALCON_SKIP_BISON         - ON to avoid using bison.
# FALCON_WITH_MANPAGES      - ON to build and install man pages.
# FALCON_WITHOUT_FEATHERS   - ON to not build feathers.
# FALCON_WITHOUT_MODULES    - ON to not build other modules.
#
##  Modules
#
# FALCON_WANT_*MODULE_NAME* - ON to build a given module
#
# List of available modules: CURL DBI DBUS DYNLIB GD2 GTK PDF SDL
#

cmake_minimum_required(VERSION 2.8)
project(Falcon_Core)

option( FALCON_SKIP_BISON "Skip BISON to avoid recompiling grammar" ON )
option( FALCON_WITHOUT_FEATHERS "Do not build feathers" OFF )

if(WIN32)
    option( FALCON_WITH_MANPAGES "Build and install also manpages" OFF )
else()
    # Manpages are mandatory on unix
    option( FALCON_WITH_MANPAGES "Build and install also manpages" ON )
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# NOTE modules are installed via
#   install(FILES .. DESTINATION ${FALCON_MOD_INSTALL_DIR})
# since they are neither RUNTIME, LIBRARY nor ARCHIVE.

#In windows, we normally install in c:\falcon
if(WIN32)
   #mingw requires -mthreads global option
   if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
      message( "MINGW make detected, adding -mthreads flag" )
      list(APPEND CMAKE_EXE_LINKER_FLAGS -mthreads )
      list(APPEND CMAKE_SHARED_LINKER_FLAGS -mthreads )
      list(APPEND CMAKE_MODULE_LINKER_FLAGS -mthreads )
   endif()
endif(WIN32)
#
## </Subset of falcon-config.cmake>

if(WIN32)
   SET( FALCON_HOST_SYSTEM "WINDOWS" )
   SET( FALCON_SYSTEM_WIN 1 )
else()
   if(APPLE)
      set( FALCON_HOST_SYSTEM "MAC" )
      set( FALCON_SYSTEM_MAC 1 )
   elseif(UNIX)
      set( FALCON_HOST_SYSTEM "UNIX" )
      SET( FALCON_SYSTEM_UNIX 1 )
   else()
      message(FATAL_ERROR "Sorry, can't determine system type" )
   endif()
endif()

#SONAME and soversion (unix so library informations for engine)
# Remember that SONAME never follows project versioning, but
# uses a VERSION, REVISION, AGE format, where
# VERSION: generational version of the project
# REVISION: times this version has been touched
# AGE: Number of version for which binary compatibility is granted
# In eample, 1.12.5 means that this lib may be dynlinked against
# every program using this lib versioned from 1.8 to 1.12.
include(versioninfo.cmake)
if(NOT FALCON_SONAME_AGE)
   # A couple of useful shortcuts
   set(FALCON_SONAME "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}.${FALCON_SONAME_AGE}")
   set(FALCON_SONAME_REV "${FALCON_SONAME_VERSION}.${FALCON_SONAME_REVISION}")
endif(NOT FALCON_SONAME_AGE)

#Automatically generated version info for RC scripts and sources
#CMAKE is good at this, let's use this feature
set(FALCON_VERSION_RC   "${FALCON_VERSION_MAJOR}, ${FALCON_VERSION_MINOR}, ${FALCON_VERSION_REVISION}, ${FALCON_VERSION_PATCH}")
set(FALCON_VERSION_ID   "${FALCON_VERSION_MAJOR}.${FALCON_VERSION_MINOR}.${FALCON_VERSION_REVISION}.${FALCON_VERSION_PATCH}")
set(FALCON_ID   "${FALCON_VERSION_MAJOR}.${FALCON_VERSION_MINOR}.${FALCON_VERSION_REVISION}")

message(STATUS "Compiling Falcon ${FALCON_VERSION_ID} on ${CMAKE_SYSTEM}" )

##############################################################################
#  Other defaults
##############################################################################
include(TestBigEndian)

message(STATUS "Testing endianity on ${CMAKE_SYSTEM}" )
TEST_BIG_ENDIAN(falcon_big_endian)
if(falcon_big_endian)
   set(FALCON_LITTLE_ENDIAN 0)
else(falcon_big_endian)
   set(FALCON_LITTLE_ENDIAN 1)
endif(falcon_big_endian)


# install prefix defaults, if not set
if(NOT CMAKE_INSTALL_PREFIX)
  #In windows, we normally install in c:\falcon
  if(WIN32)
    if($ENV{PRGORAMS})
      SET(CMAKE_INSTALL_PREFIX  "C:\\\\$ENV{PROGRAMS}\\\\falcon" )
    else()
      SET(CMAKE_INSTALL_PREFIX  "C:\\\\Program Files\\\\falcon" )
    endif()
  else()
    set(CMAKE_INSTALL_PREFIX  "/usr/local" )
  endif()
endif()

message( STATUS "Installation prefix: ${CMAKE_INSTALL_PREFIX}" )

if (NOT FALCON_LIB_DIR)
   set(FALCON_LIB_DIR lib)
endif()

if( NOT FALCON_SHARE_DIR)
   if(WIN32)
      set(FALCON_SHARE_DIR "share")
   else()
      set(FALCON_SHARE_DIR "share/falcon${FALCON_ID}")
   endif()
endif()

if (NOT FALCON_BIN_DIR)
   set(FALCON_BIN_DIR bin)
endif()

if (NOT FALCON_INC_DIR)
   if(WIN32)
      set(FALCON_INC_DIR "include")
   else()
      set(FALCON_INC_DIR "include/falcon${FALCON_ID}")
   endif()
endif()

if (NOT FALCON_MOD_DIR )
   if(WIN32)
      set(FALCON_MOD_DIR bin)
   else()
      set(FALCON_MOD_DIR "${FALCON_LIB_DIR}/falcon")
   endif()
endif()

if(WIN32)
  set(FALCON_CMAKE_DIR cmake)
else()
  set(FALCON_CMAKE_DIR ${FALCON_SHARE_DIR}/cmake)
endif()

set(FALCON_APP_DIR ${FALCON_MOD_DIR}/apps)

# for install(TARGETS .. ${FALCON_INSTALL_DESTINATIONS})
set(FALCON_INSTALL_DESTINATIONS
  RUNTIME DESTINATION ${FALCON_BIN_DIR}
  LIBRARY DESTINATION ${FALCON_LIB_DIR}
  ARCHIVE DESTINATION ${FALCON_LIB_DIR}
)

message( STATUS "Binary prefix: ${FALCON_BIN_DIR}" )
message( STATUS "Library prefix: ${FALCON_LIB_DIR}" )
message( STATUS "Include prefix: ${FALCON_INC_DIR}" )
message( STATUS "Module prefix: ${FALCON_MOD_DIR}" )
message( STATUS "Application directory: ${FALCON_APP_DIR}" )
message( STATUS "CMAKE config prefix: ${FALCON_CMAKE_DIR}" )

if ( NOT WIN32 )
   if (NOT FALCON_MAN_DIR)
      set(FALCON_MAN_DIR "share/man/man1")
   endif()
   message( STATUS "Manual pages: ${FALCON_MAN_DIR}" )
endif()


#########################################################à
# List of files to be installed as docs
#

set( doc_files
      AUTHORS
      BUILDING
      ChangeLog
      copyright
      LICENSE
      README
      README.editline
      README.mersenne
      TODO
      LICENSE_GPLv2
      RELNOTES
   )

install(
   FILES ${doc_files}
   DESTINATION ${FALCON_SHARE_DIR} )

#########################################################à
# Subdirectories
#

#collect the include files in FALCON_HEADER
add_subdirectory(include)
#
add_subdirectory(engine)
add_subdirectory(clt)
add_subdirectory(devtools)


# and man pages
if (FALCON_WITH_MANPAGES)
   add_subdirectory(manpages)
endif()

#########################################################################
# Feathers?
#
if( NOT FALCON_WITHOUT_FEATHERS )
   message( STATUS "Adding FEATHERS to this build" )

   #override settings for the include dirs
   set( Falcon_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/include" )
   list( APPEND Falcon_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" )

   add_subdirectory( feathers )

else()
    message( STATUS "We are not compiling feathers" )
endif()

#########################################################################
# Modules?
#
if( NOT FALCON_WITHOUT_MODULES )
    message( STATUS "Adding all modules to this build" )

    option( FALCON_WANT_CURL "Compile curl module" ON )
    option( FALCON_WANT_DBI "Compile DBI module" ON )
    option( FALCON_WANT_DBUS "Compile DBUS module" ON )
    option( FALCON_WANT_DYNLIB "Compile DynLib module" ON )
    option( FALCON_WANT_GD2 "Compile GD2 module" ON )
    option( FALCON_WANT_GTK "Compile Gtk+ module" ON )
    option( FALCON_WANT_PDF "Compile PDF module" ON )
    option( FALCON_WANT_SDL "Compile SDL module" ON )

else()
    message( STATUS "We are not compiling any modules" )

    option( FALCON_WANT_CURL "Compile curl module" OFF )
    option( FALCON_WANT_DBI "Compile DBI module" OFF )
    option( FALCON_WANT_DBUS "Compile DBUS module" OFF )
    option( FALCON_WANT_DYNLIB "Compile DynLib module" OFF )
    option( FALCON_WANT_GD2 "Compile GD2 module" OFF )
    option( FALCON_WANT_GTK "Compile Gtk+ module" OFF )
    option( FALCON_WANT_PDF "Compile PDF module" OFF )
    option( FALCON_WANT_SDL "Compile SDL module" OFF )

endif()
add_subdirectory( modules )

#########################################################################
# Tests?

option(FALCON_WITH_CTEST_TESTS "" off)
if(FALCON_WITH_CTEST_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# CMake generated information.  Is used by our falcon-config.cmake
install(EXPORT falcon-core-targets
  DESTINATION ${FALCON_CMAKE_DIR}
)
