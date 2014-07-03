#
# Configurator for development tools
#

# CMake will find it via find_package(falcon [REQUIRED])
# It supplies include paths, libraries (via falcon-core-targets.cmake)
# and convenience MACROS.
#

set( cmake_mod_files
   FalconConfig.cmake
   falcon-core-config.cmake
   falcon-core-config-version.cmake )
   
foreach(item ${cmake_mod_files} )
  message( "Configuring ${item}" )
  configure_file(
    ${item}.in  "${CMAKE_CURRENT_BINARY_DIR}/${item}"
    ESCAPE_QUOTES @ONLY)

    #Create installation files from in files
    list( APPEND cmake_mod_files_inst ${CMAKE_CURRENT_BINARY_DIR}/${item} )
endforeach()

#
# Configure other binary/executable files
#

IF(WIN32)
   SET(sys_in_spec)
ELSE()
   SET(sys_in_spec falcon-conf)
ENDIF()

SET(in_bin_files
  falconeer.fal
  ${sys_in_spec}
)

foreach(item ${in_bin_files} )
  message( "Configuring ${item}" )
  configure_file(
    ${item}.in  "${CMAKE_CURRENT_BINARY_DIR}/${item}"
    ESCAPE_QUOTES @ONLY)
    
    #Create installation files from in files
    list( APPEND inst_bin_files "${CMAKE_CURRENT_BINARY_DIR}/${item}" )
endforeach()

#
# Add non-configured files to be installed
#

########################################################################
# Install

install(
  FILES ${inst_bin_files}
  DESTINATION ${FALCON_BIN_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE
)

install(
   FILES ${cmake_mod_files_inst} 
   DESTINATION ${FALCON_CMAKE_DIR}
)
