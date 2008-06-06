#
# Just install include files so that every part of the project can use it.
#

CONFIGURE_FILE(
         falcon/config.h.in
         "${FALCON_BUILD_TREE}/core/include/falcon/config.h"
         ESCAPE_QUOTES
         @ONLY )

FILE( GLOB fasm_files "fasm/*.h" )
FILE( GLOB falcon_files "falcon/*.h" )

INSTALL( FILES ${fasm_files}
   DESTINATION "${FALCON_INC_DIR}/fasm")

INSTALL( FILES ${falcon_files}
   DESTINATION "${FALCON_INC_DIR}/falcon")

#Fixes a small bug of CMAKE in FILES install rule.
IF(WIN32)
   STRING(REPLACE "\\" "/" FALCON_BUILD_TREE "${FALCON_BUILD_TREE}" )
ENDIF(WIN32)

INSTALL( FILES "${FALCON_BUILD_TREE}/core/include/falcon/config.h"
   DESTINATION "${FALCON_INC_DIR}/falcon")

