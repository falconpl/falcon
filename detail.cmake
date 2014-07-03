#################################################################
# Base options
option( FALCON_SKIP_BISON "Skip BISON to avoid recompiling grammar" ON )
option( FALCON_BUILD_FEATHERS "Build Falcon feathers" ON )
option( FALCON_BUILD_MODULES "Build Falcon modules" ON )
option( FALCON_BUILD_FWKS "Build Falcon frameworks" ON )
option( FALCON_BUILD_APPS "Build Falcon applications" ON )
option( FALCON_BUILD_NATMODS "Build native (binary) non-feather modules" ON )
option( FALCON_BUILD_DOCS "Build automatic documentation" ON )
option( FALCON_BUILD_BINTESTS "Build embedding and other binary tests" OFF )
option( FALCON_INSTALL_TESTS "Copy test files in the final installation (under share/)" OFF )
option( FALCON_BUILD_DIST "Prepare distribution helper scripts in dist/" OFF )
option( FALCON_TRACE_GC "Add support to trace GC operations at runtime" ON )

if (WIN32)
	set( FALCON_COMPILE_SOURCE_MODS OFF )
	set( FALCON_STRIP_SOURCE_MODS OFF)
else()
	option( FALCON_COMPILE_SOURCE_MODS "Compile source modules into .fam for faster script startup" ON )
	option( FALCON_STRIP_SOURCE_MODS "Don't install source .fal/ftd modules" OFF)
endif()

# NOTE modules are installed via
#   install(FILES .. DESTINATION ${FALCON_MOD_INSTALL_DIR})
# since they are neither RUNTIME, LIBRARY nor ARCHIVE.

#In windows, we normally install in c:\falcon
if(WIN32)
   #mingw requires -mthreads global option
   if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
      if ("${CMAKE_SIZEOF_VOID_P}" EQUAL "4")
         message( "MINGW make detected, adding -march=i486 flag" )
         set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=i486" )
         set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=i486 " )
      endif()
      message( "MINGW make detected, adding -mthreads flag" )
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mthreads" )
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mthreads" )
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -mthreads" )
      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -mthreads" )
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
      add_definitions(-DUSE_CXX0X)
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
#
set( Falcon_VERSION "${FALCON_VERSION_ID}" )

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
#
if(LIB_SUFFIX)
  set(FALCON_LIB_DIR "${FALCON_LIB_DIR}${LIB_SUFFIX}")
endif()

if( NOT FALCON_SHARE_DIR)
   if(WIN32)
      set(FALCON_SHARE_DIR "share")
   else()
      set(FALCON_SHARE_DIR "share/falcon${FALCON_ID}")
   endif()
endif()

if( NOT FALCON_DOC_DIR)
   if(WIN32)
      set(FALCON_DOC_DIR "share")
   else()
      set(FALCON_DOC_DIR "share/doc/falcon${FALCON_ID}")
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

# Variable Forwarding, interally we use the conventions
# introduced by find_package(Falcon). TODO: That's not elegant
set(Falcon_APP_DIR "${FALCON_APP_DIR}")
set(Falcon_MOD_DIR "${FALCON_MOD_DIR}")
set(Falcon_BIN_DIR "${FALCON_BIN_DIR}")
set(Falcon_LIB_DIR "${FALCON_LIB_DIR}")
set(Falcon_MAN_DIR "${FALCON_MAN_DIR}")
set(Falcon_INC_DIR "${FALCON_INC_DIR}")
set(Falcon_SHARE_DIR "${FALCON_SHARE_DIR}")
set(Falcon_DOC_DIR "${FALCON_DOC_DIR}")
set(Falcon_CMAKE_DIR "${FALCON_CMAKE_DIR}")


#########################################################################
# RPATH(Linux) and install_name(OSX)
#
option(DISABLE_RPATH "http://wiki.debian.org/RpathIssue" OFF)
if(NOT DISABLE_RPATH)
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

function( add_fam_target source )
   file( RELATIVE_PATH source_relative "${CMAKE_SOURCE_DIR}" "${source}")
   
   get_filename_component( path_of_fal "${source_relative}"  PATH)
   get_filename_component( name_of_fal "${source_relative}"  NAME_WE)

   # falcon command -- on windows it
   if(UNIX OR APPLE)
      set( falcon_command "${CMAKE_BINARY_DIR}/devtools/icomp.sh" )
   else()
      set( falcon_command "${CMAKE_BINARY_DIR}/devtools/icomp.bat" )
   endif()

   set( output_file "${CMAKE_BINARY_DIR}/${path_of_fal}/${name_of_fal}.fam" )
   set( compile_command ${falcon_command} ${source} ${output_file} )

   add_custom_command(
      OUTPUT "${output_file}"
      COMMAND ${compile_command}
      DEPENDS ${source}
   )
   
   string(REPLACE "/" "_" target_name "${source}" )
         
   add_custom_target(${target_name} ALL DEPENDS "${output_file}" falcon falcon_engine )

   #install must be relative to current source path_of_fal
   file( RELATIVE_PATH single_fal_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${single_fal}")
   get_filename_component( path_of_fal "${single_fal_relative}"  PATH)
   install(FILES "${output_file}" DESTINATION "${FALCON_MOD_DIR}/${path_of_fal}")

endfunction()

function( falcon_install_moddirs module_dirs )

   message( "Installing top modules in ${CMAKE_CURRENT_SOURCE_DIR}" )
   
   foreach(item ${module_dirs} )
      message( "Installing falcon modules in ${item}" )
      file( GLOB_RECURSE files "${item}" "*.fal" "*.ftd" )
      foreach( single_fal ${files} )
         file( RELATIVE_PATH single_fal_relative "${CMAKE_CURRENT_SOURCE_DIR}" "${single_fal}")
         get_filename_component( path_of_fal "${single_fal_relative}"  PATH)

         #Create installation files from in files
         if(NOT FALCON_STRIP_SOURCE_MODS)
            install(
               FILES "${single_fal}"
               DESTINATION "${FALCON_MOD_DIR}/${path_of_fal}"
            )
         endif()
         
         if(FALCON_COMPILE_SOURCE_MODS)
            add_fam_target( ${single_fal} )
         endif()
      endforeach()
   endforeach()
   
endfunction()

# vi: set ai et sw=3 sts=3:

