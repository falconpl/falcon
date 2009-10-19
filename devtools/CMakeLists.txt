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
  configure_file(
    ${item}.in  ${CMAKE_CURRENT_BINARY_DIR}/${item}
    ESCAPE_QUOTES @ONLY)
endforeach(item)

SET(inst_files
  "${CMAKE_CURRENT_BINARY_DIR}/falconeer.fal"
  "fallc.fal"
  ${sys_inst_spec}
)


INSTALL(
  FILES ${inst_files}
  DESTINATION bin
  PERMISSIONS OWNER_READ OWNER_WRITE 
              OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE
              WORLD_READ WORLD_EXECUTE
)
