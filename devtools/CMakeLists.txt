#
# Configurator for development tools
#

#
# Configure other binary/executable files
#

IF(WIN32)
   SET(sys_in_spec)
ELSE()
   SET(sys_in_spec falcon-conf)
ENDIF()

SET(in_bin_files  
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
