#################################################################
#
#   FALCON - The Falcon Programming Language.
#   FILE: /cmake/utils.cmake
#
#   Utilities for the cmake build process of Falcon engine.
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

#################################################################
# Adds a precompiled falcon scripts as a target.
# source - the fal/ftd file that generates the module
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


#################################################################
# Installs directories containing falcon source files.
# module_dirs - a list of directories conaining fal/ftd
#               source Falcon modules to be installed.
#

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


#################################################################
# Create a publicly visible option with a default value
# OPT - The option
# desc - The description of the option 
# VAL - ON or OFF

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


#################################################################
# Create a publicly visible variable with a default value
# OPT -  The variable name
# desc - The description of the variable
# VAL - the default value for the variable

function( set_default_opt OPT desc VAL )
	option( ${OPT} desc ${VAL} )
	message( STATUS "FALCON_${OPT}{${VAL}} - ${desc} [ON|OFF]" )
endfunction()



#################################################################
# Automatic binding module generation
# module - The name of a falcon module
# lib - the CMAKE name for the bound library 
# desc - description to be sent to message

macro( FalconMakeBinding module lib desc )

   if( FALCON_NATMODS_AUTO )
      find_package( ${lib} )
      if ( ${lib}_FOUND )
         set_default_opt( FALCON_BUILD_${module} "${desc}" ON )
      else()
         set_default_opt( FALCON_BUILD_${module} "${desc}" OFF )
      endif()
   else()
      set_default_opt( FALCON_BUILD_${module} "${desc}" OFF )
   endif()

endmacro()
