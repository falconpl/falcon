#
# Just install include files so that every part of the project can use it.
#

FILE( GLOB fasm_files "fasm/*.h" )
FILE( GLOB falcon_files "falcon/*.h" )

INSTALL( FILES ${fasm_files}
   DESTINATION ${TARGET_DIR}/include/fasm)

INSTALL( FILES ${falcon_files}
   DESTINATION ${TARGET_DIR}/include/falcon)

CONFIGURE_FILE(
         falcon/config.h.in
         ${FALCON_BUILD_TREE}/include/falcon/config.h
         ESCAPE_QUOTES
         @ONLY )

# This copy should actually be performed during install,
# but a bug in current windows version of CMAKE (2.4.7)  prevents it.
CONFIGURE_FILE(
         falcon/config.h.in
         ${TARGET_DIR}/include/falcon/config.h
         ESCAPE_QUOTES
         @ONLY )