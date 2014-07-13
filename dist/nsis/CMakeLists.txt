#
# Configurator for NSIS installer
#
cmake_minimum_required(VERSION 2.6)

#
# Files to be configured
#
SET( config_files

  falcon_inst.nsi
)

# Files to be copied
SET( copy_files

	fileassoc.nsh
	WriteEnvStr.nsh
	AddToPath.nsh
)

# Directories to be copied
SET( copy_directories
		
	Resources
)


##################################################
# Prepare configure-time variables for NSIS script
#

if( "${CMAKE_GENERATOR}" STREQUAL "MinGW Makefiles" )
   set( FALCON_ENGINE_DLL "libFalcon_engine.dll" )
   set( FALCON_ENGINE_LIB "libFalcon_engine.dll.a" )
   
   find_library(MINGW_MD_RUNTIME mingwm10)
   if( NOT "${MINGW_MD_RUNTIME}" STREQUAL "MINGW_MD_RUNTIME-NOTFOUND" )
      get_filename_component(_filename ${MINGW_MD_RUNTIME} NAME)
      configure_file( ${MINGW_MD_RUNTIME} ${Falcon_BINARY_DIR}/${_filename} COPYONLY)
   endif() 
  
   set( FALCON_BUILD_NAME "MinGW" )
   
else()
   set( FALCON_ENGINE_DLL "Falcon_engine.dll" )
   set( FALCON_ENGINE_LIB "Falcon_engine.lib" )
   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 10" )
	  set( FALCON_BUILD_NAME "VS10" )
   endif()
   
   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 10 Win64" )
	  set( FALCON_BUILD_NAME "VS10_64" )
   endif()
   
   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 6" )
	  set( FALCON_BUILD_NAME "VS6" )
   endif()

   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 8 2005" )
	  set( FALCON_BUILD_NAME "VS8" )
   endif()

   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 8 2005 Win64" )
	  set( FALCON_BUILD_NAME "VS8_64" )
   endif()
   
   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 9 2008" )
	  set( FALCON_BUILD_NAME "VS9" )
   endif()

   if( "${CMAKE_GENERATOR}" STREQUAL "Visual Studio 9 2008 Win64" )
	  set( FALCON_BUILD_NAME "VS9_64" )
   endif()
	
   if(NOT FALCON_BUILD_NAME )
	  set( FALCON_BUILD_NAME  "Win32" )
   endif()

endif()


if ( ${FALCON_VERSION_REVISION} EQUAL 0 )
   if ( ${FALCON_VERSION_PATCH} EQUAL 0 )
      set( FALCON_VF_NAME  
         "${FALCON_VERSION_MAJOR}_${FALCON_VERSION_MINOR}_${FALCON_BUILD_NAME}" )
   else()
      set( FALCON_VF_NAME  
         "${FALCON_VERSION_MAJOR}_${FALCON_VERSION_MINOR}_${FALCON_VERSION_REVISION}_${FALCON_VERSION_PATCH}" )
   endif()
else()
   set( FALCON_VF_NAME  
      "${FALCON_VERSION_MAJOR}_${FALCON_VERSION_MINOR}_${FALCON_VERSION_REVISION}_${FALCON_VERSION_PATCH}_${FALCON_BUILD_NAME}" )
endif()

################################################
# Execute what required
#

foreach( item ${config_files} )
  message( "Configuring ${item}" )
  configure_file(
    ${item}.in  "${CMAKE_CURRENT_BINARY_DIR}/${item}"
    ESCAPE_QUOTES @ONLY)
    
    #Create installation files from in files
    list( APPEND inst_files "${CMAKE_CURRENT_BINARY_DIR}/${item}" )
endforeach()

foreach( item ${copy_files} )
  message( "Copying ${item}" )
  list( APPEND inst_files "${CMAKE_CURRENT_SOURCE_DIR}/${item}" )
endforeach()

########################################################################
# Install

install(
  FILES ${inst_files}
  DESTINATION dist
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE
)

foreach( item ${copy_directories} )
	install(
		DIRECTORY ${item}
		DESTINATION dist
	)
endforeach()
