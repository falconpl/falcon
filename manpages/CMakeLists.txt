#
# Install all the manual files.
#

cmake_minimum_required(VERSION 2.4)
SET(MAN_SRC_DIR "$ENV{FALCON_SRC_TREE}/docs/manpages/man1")
SET(MAN_BUILD_DIR $ENV{FALCON_BUILD_TREE}/share/man/man1)

#A different install tree may be given by the installation script.
IF( "${FALCON_INSTALL_TREE}" STREQUAL "" )
   SET( FALCON_INSTALL_TREE  "$ENV{FALCON_ACTIVE_TREE}" )
ENDIF( "${FALCON_INSTALL_TREE}" STREQUAL "" )

SET(MAN_TARGET_DIR ${FALCON_INSTALL_TREE}/share/man/man1)

EXECUTE_PROCESS(COMMAND mkdir -p ${MAN_BUILD_DIR})

MACRO( FALCON_ADD_ZIPS list )
   FOREACH( item ${${list}} )
      ADD_CUSTOM_TARGET( ${item}_manpage ALL
               COMMAND gzip --best ${MAN_SRC_DIR}/${item}.1 -c > ${MAN_BUILD_DIR}/${item}.1.gz
               VERBATIM )
   ENDFOREACH( item ${${list}} )
ENDMACRO( FALCON_ADD_ZIPS )

SET( man_files
  falcon  falconeer.fal fallc.fal  faldisass  falrun  faltest falcon-conf )
FALCON_ADD_ZIPS( man_files )

INSTALL( FILES ${MAN_BUILD_DIR}/falcon.1.gz
            ${MAN_BUILD_DIR}/falconeer.fal.1.gz
            ${MAN_BUILD_DIR}/fallc.fal.1.gz
            ${MAN_BUILD_DIR}/faldisass.1.gz
            ${MAN_BUILD_DIR}/falrun.1.gz
            ${MAN_BUILD_DIR}/faltest.1.gz
            ${MAN_BUILD_DIR}/falcon-conf.1.gz
   DESTINATION ${MAN_TARGET_DIR} )
