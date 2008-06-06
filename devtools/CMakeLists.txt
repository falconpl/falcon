#
# Configurator for development tools
#

IF(UNIX)
   SET( sys_in_spec falcon-conf )
   SET( sys_inst_spec "${FALCON_BUILD_TREE}/core/devtools/falcon-conf" )
ENDIF(UNIX)

#Fixes a small bug of CMAKE in FILES install rule.
IF(WIN32)
   STRING(REPLACE "\\" "/" FALCON_BUILD_TREE "${FALCON_BUILD_TREE}" )
ENDIF(WIN32)

SET( in_files
   falconeer.fal
   ${sys_in_spec}
   )

SET( inst_files
   "${FALCON_BUILD_TREE}/core/devtools/falconeer.fal"
   "fallc.fal"
   ${sys_inst_spec}
   )

FOREACH(item ${in_files} )
   CONFIGURE_FILE(
      ${item}.in
      "${FALCON_BUILD_TREE}/core/devtools/${item}"
      ESCAPE_QUOTES
      @ONLY )
ENDFOREACH(item)

INSTALL(FILES ${inst_files}
          DESTINATION ${FALCON_BIN_DIR}
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE )
