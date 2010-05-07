#
# Configurator for development tools
#

IF(UNIX)
   SET(sys_in_spec falcon-conf )
   SET(sys_inst_spec ${CMAKE_CURRENT_BINARY_DIR}/falcon-conf)
ENDIF(UNIX)

SET(in_files
  falconeer.fal
  ${sys_in_spec}
)


foreach(item ${in_files} )
  message( "Configuring ${item}" )
  configure_file(
    ${item}.in  ${CMAKE_CURRENT_BINARY_DIR}/${item}
    ESCAPE_QUOTES @ONLY)
endforeach(item)

# CMake will find it via find_package(falcon [REQUIRED])
# It supplies include paths, libraries (via falcon-core-targets.cmake)
# and convenience MACROS.
#
message( "Configuring FalconConfig.cmake" )
configure_file(
  FalconConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FalconConfig.cmake
  @ONLY
)


########################################################################
# Install

SET(inst_files
  "${CMAKE_CURRENT_BINARY_DIR}/falconeer.fal"
  "fallc.fal"
  ${sys_inst_spec}
)


INSTALL(
  FILES ${inst_files}
  DESTINATION ${FALCON_BIN_DIR}
  PERMISSIONS OWNER_READ OWNER_WRITE 
              OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE
)

install(FILES
   ${CMAKE_CURRENT_BINARY_DIR}/FalconConfig.cmake
   ${CMAKE_CURRENT_BINARY_DIR}/modfunctions.cmake
   DESTINATION ${FALCON_CMAKE_DIR}
)
configure_file(modfunctions.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/modfunctions.cmake @ONLY)