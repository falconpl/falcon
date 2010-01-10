####################################################################
# The Falcon Programming language
#
# CMake configuration file for Core falcon
####################################################################
cmake_minimum_required(VERSION 2.6)
project(Falcon_Core)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

# NOTE The following is alos content of falcon-config.make.in.  We might 
# seperate it into its own file and generate 
#   ${CMAKE_BINARY_DIR}/falcon-conf.cmake 
# our self rather then with configure_file.  For now its to short for 
# this hassle.
#
## <Subset of falcon-config.cmake.in>
#

if (NOT FALCON_LIB_DIR)
   set(FALCON_LIB_DIR lib)
endif()

if (NOT FALCON_MOD_INSTALL_DIR)
if(WIN32)
   set(FALCON_MOD_INSTALL_DIR bin)
else()
   set(FALCON_MOD_INSTALL_DIR "${FALCON_LIB_DIR}/falcon")
endif()
endif()

# for install(TARGETS .. ${FALCON_INSTALL_DESTINATIONS})
set(FALCON_INSTALL_DESTINATIONS
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION ${FALCON_LIB_DIR} 
  ARCHIVE DESTINATION ${FALCON_LIB_DIR}
)
# NOTE modules are installed via 
#   install(FILES .. DESTINATION ${FALCON_MOD_INSTALL_DIR})
# since they are neither RUNTIME, LIBRARY nor ARCHIVE. 

#In windows, we normally install in c:\falcon
if(WIN32)
   #mingw requires -mthreads global option
   if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
      list(APPEND CMAKE_EXE_LINKER_FLAGS -mthreads )
      list(APPEND CMAKE_SHARED_LINKER_FLAGS -mthreads )
      list(APPEND CMAKE_MODULE_LINKER_FLAGS -mthreads )
   endif()
endif(WIN32)
#
## </Subset of falcon-config.cmake>

option(DISABLE_RPATH "http://wiki.debian.org/RpathIssue" off)
if(NOT DISABLE_RPATH)
  # Always find libfalcon_engone.so in build and install tree, without LD_LIBRARY_PATH.
  set(CMAKE_SKIP_BUILD_RPATH  false)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH false) 
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${FALCON_LIB_DIR}")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)
endif()

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


message(STATUS "Compiling Falcon on ${CMAKE_SYSTEM}" )

##############################################################################
#  Other defaults
##############################################################################
include(TestBigEndian)

message(STATUS "Testing endianity on ${CMAKE_SYSTEM}" )
test_big_endian(falcon_big_endian)
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

if(NOT FALCON_FINAL_DESTINATION)
   SET(FALCON_FINAL_DESTINATION $CMAKE_INSTALL_PREFIX)
endif()

#
add_subdirectory(engine)
add_subdirectory(clt)
add_subdirectory(devtools)

#also performs installation of the include directory
add_subdirectory(include)

option(FALCON_WITH_CTEST_TESTS "" off)
if(FALCON_WITH_CTEST_TESTS)
  enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()


configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/falcon-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/falcon-config.cmake
  @ONLY
)
# CMake will find it via find_package(falcon [REQUIRED])
# It supplies include paths, libraries (via falcon-core-targets.cmake)
# and convenience MACROS. 
#
# TODO We add the -mthread flag to MinGW builds without asking the 
# module developer, is this ok?
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/falcon-config.cmake
  DESTINATION lib/falcon/cmake
)

# CMake generated information.  Is used by our falcon-config.cmake
install(EXPORT falcon-core-targets
  DESTINATION lib/falcon/cmake
)
