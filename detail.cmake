#################################################################
# Base options
option( FALCON_SKIP_BISON "Skip BISON to avoid recompiling grammar" ON )
option( FALCON_BUILD_FEATHERS "Build Falcon feathers" ON )
option( FALCON_BUILD_MODULES "Build Falcon modules" ON )
option( FALCON_BUILD_FWKS "Build Falcon frameworks" ON )
option( FALCON_BUILD_APPS "Build Falcon applications" ON )
option( FALCON_BUILD_NATMODS "Build native (binary) non-feather modules" ON )
option( FALCON_COMPILE_FAMS "Compile falcon modules when installing them" ON )
option( FALCON_STRIP_FALS "Remove source modules when installing" OFF )

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

## SONAME and soversion (unix so library informations for engine)
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
    if($ENV{PROGRAMS})
      SET(CMAKE_INSTALL_PREFIX  "C:\\\\$ENV{PROGRAMS}\\\\falcon" )
    else()
      SET(CMAKE_INSTALL_PREFIX  "C:\\\\Program Files\\\\falcon" )
    endif()
  else() # unixes
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


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${FALCON_BIN_DIR}" CACHE INTERNAL 
  "Where to put the executables"
)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${FALCON_LIB_DIR}" 
  CACHE INTERNAL "Where to put the libraries"
)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${FALCON_LIB_DIR}" CACHE INTERNAL 
  "Where to put the archives"
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

option(DISABLE_RPATH "http://wiki.debian.org/RpathIssue" on)
if(NOT DISABLE_RPATH)
  message(FATAL_ERROR)
  # Always find libfalcon_engine.so in build and install tree, without LD_LIBRARY_PATH.
  set(CMAKE_SKIP_BUILD_RPATH  false)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH false)
  set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${FALCON_LIB_DIR}")
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH false)

  # Apple equivalent to RPATH is called `install_name'
  set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${FALCON_LIB_DIR}")

else()
  set(CMAKE_SKIP_RPATH on)
endif()

#########################################################################
# Functions
#

function( falcon_install_moddirs module_dirs )

   message( "Installing top modules in ${CMAKE_CURRENT_SOURCE_DIR}" )
   file( GLOB modules "*.fal" )
   foreach( single_fal ${modules} )
      install(
         FILES ${single_fal}
         DESTINATION ${FALCON_MOD_DIR}
      )
   endforeach()

   foreach(item ${module_dirs} )
      message( "Installing falcon modules in ${item}" )
      file( GLOB_RECURSE files "${item}" "*.fal" "*.ftd" )
      foreach( single_fal ${files} )
         file( RELATIVE_PATH single_fal_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${single_fal}")
         get_filename_component( path_of_fal "${single_fal_relative}"  PATH)

         #Create installation files from in files
         install(
            FILES "${single_fal}"
            DESTINATION "${FALCON_MOD_DIR}/${path_of_fal}"
         )
      endforeach()


   endforeach()

endfunction()