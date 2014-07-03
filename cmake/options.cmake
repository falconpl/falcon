#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/options.cmake
#
#   Project-level options.
#   -------------------------------------------------------------------
#   Author: Giancarlo Niccolai
#   Begin: Wed, 02 Jul 2014 23:02:17 +0200
#
#   -------------------------------------------------------------------
#   (C) Copyright 2014: the FALCON developers (see list in AUTHORS file)
#
#   See LICENSE file for licensing details.
#   
#################################################################


function( set_default_var OPT desc VAL )
	set(NAME "FALCON_${OPT}" )
	if("${${NAME}}" STREQUAL "" )
		set("${NAME}" "${VAL}" PARENT_SCOPE)
		set("${NAME}" "${VAL}" )
	endif()
	Message( STATUS "${NAME}{${${NAME}}} - ${desc}" )
	
	set(lowname "Falcon_${OPT}")
	set("${lowname}" "${VAL}" PARENT_SCOPE)
	#message( STATUS "(lowercase) ${lowname} = ${${lowname}}" )
endfunction()


function( set_default_opt OPT desc VAL )
	option( ${OPT} desc ${VAL} )
	message( STATUS "FALCON_${OPT}{${VAL}} - ${desc} [ON|OFF]" )
endfunction()



Message("#################################################################")
Message("#                 TOP-LEVEL BUILD OPTIONS                       #")
Message("#################################################################")

Message("Overall build options: ")
set_default_opt(DISABLE_RPATH "Disable RPATH management - http://wiki.debian.org/RpathIssue" OFF)

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
 
message( STATUS "CMAKE_INSTALL_PREFIX{${CMAKE_INSTALL_PREFIX}} - Installation prefix" )


Message("Build set selection options: ")
set_default_opt( FALCON_BUILD_NATIVE_MODS "Build native (binary) non-feather modules" ON )
set_default_opt( FALCON_BUILD_SOURCE_MODS "Compile source modules into .fam for faster script startup" ON )
set_default_opt( FALCON_STRIP_SOURCE_MODS   "Don't install source .fal/ftd modules" OFF)

set_default_opt( FALCON_BUILD_FWKS "Build Falcon frameworks" ON )
set_default_opt( FALCON_BUILD_APPS "Build Falcon applications" ON )

set_default_opt( FALCON_BUILD_DOCS "Build automatic documentation" ON )
set_default_opt( FALCON_BUILD_BINTESTS "Build embedding and other binary tests" OFF )


Message("Build installation options: ")

set_default_opt( FALCON_INSTALL_TESTS "Copy test files in the final installation (under share/)" OFF )
set_default_opt( FALCON_BUILD_DIST "Prepare distribution helper scripts in dist/" OFF )

Message("Build mode options: ")

set_default_opt( FALCON_STATIC_ENGINE "Perform a static compilation of the falcon engine" OFF )
set_default_opt( FALCON_STATIC_FEATHERS  "Perform a static compilation of the main modules" OFF )
set_default_opt( FALCON_STATIC_MODS  "Perform a static compilation of the non-feathers canonical modules" OFF )

Message("Debug options: ")

set_default_opt( FALCON_TRACE_GC "Add support to trace GC operations at runtime" ON )
if( FALCON_TRACE_GC )
   # This variable is set in configuration
   set( FALCON_TRACE_GC_VALUE 1 )
else()
   set( FALCON_TRACE_GC_VALUE 0 )
endif()


#################################################################
# Setting the default for control variables
#

Message("Target directory options: ")

set_default_var(BIN_DIR "Prefix for installation of binary executables" "bin")
set_default_var(LIB_DIR "Prefix for installation of libraries" "lib")

if(WIN32)
	set( default_FALCON_APP_DIR "apps")
	set( default_FALCON_INC_DIR "include" )
	set( default_FALCON_MOD_DIR "bin" )
	set( default_FALCON_SHARE_DIR "share" )
	set( default_FALCON_DOC_DIR "docs" )
else()	
	set( default_FALCON_APP_DIR "${FALCON_LIB_DIR}exec${LIB_SUFFIX}/falcon")
	set( default_FALCON_INC_DIR "include/falcon${FALCON_ID}" )
	set( default_FALCON_MOD_DIR "${FALCON_LIB_DIR}/falcon" )
	set( default_FALCON_SHARE_DIR "share/falcon${FALCON_ID}" )
endif()

set_default_var(INC_DIR "Prefix for installation of include (.h) files" "${default_FALCON_INC_DIR}")
set_default_var(APP_DIR "Prefix for installation of falcon applications" "${default_FALCON_APP_DIR}")
set_default_var(MOD_DIR "Prefix for installation of falcon modules" "${default_FALCON_MOD_DIR}")
set_default_var(SHARE_DIR "Prefix for installation of shared files" "${default_FALCON_SHARE_DIR}")

if(WIN32)
	set( default_FALCON_DOC_DIR "docs" )
	set( default_FALCON_MAN_DIR "docs" )
	set( default_FALCON_CMAKE_DIR "cmake" )
else()	
	set( default_FALCON_DOC_DIR "${FALCON_SHARE_DIR}/share/doc/falcon${FALCON_ID}" )
	set( default_FALCON_MAN_DIR "${FALCON_SHARE_DIR}/man/man1")
	set( default_FALCON_CMAKE_DIR "${FALCON_SHARE_DIR}/cmake" )	
endif()

set_default_var(DOC_DIR "Prefix for installation of documentation files" "${default_FALCON_DOC_DIR}")
set_default_var(MAN_DIR "Prefix for installation of manual files"  "${default_FALCON_MAN_DIR}")
set_default_var(CMAKE_DIR "Prefix for installation of CMAKE files" "${default_FALCON_CMAKE_DIR}")
  
if(LIB_SUFFIX)
  set(FALCON_LIB_DIR "${FALCON_LIB_DIR}${LIB_SUFFIX}")
  set(Falcon_LIB_DIR "${FALCON_LIB_DIR}")
  Message( STATUS "LIB_SUFFIX specified - Library changed to ${FALCON_LIB_DIR}" ) 
else()
  Message( STATUS "(Specify LIB_SUFFIX to add a specific system-suffix for library install prefix)" ) 
endif()
 
 
Message("#################################################################")
Message("#               END OF TOP-LEVEL BUILD OPTIONS                  #")
Message("#################################################################")

  