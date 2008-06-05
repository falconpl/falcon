##################################################
# Falcon Programming Language
#
# Falrun
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET(SYS_RC falrun.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( falrun
   falrun.cpp
   ${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(falrun falcon_engine)

#Install
INSTALL( TARGETS falrun
      DESTINATION ${FALCON_BIN_DIR} )
