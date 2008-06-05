##################################################
# Falcon Programming Language
#
# Faldisass
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET(SYS_RC faldisass.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( faldisass
   faldisass.cpp
   ${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(faldisass falcon_engine)

#Install
INSTALL( TARGETS faldisass
      DESTINATION ${FALCON_BIN_DIR} )
