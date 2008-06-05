##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYSTEM_RC falcon.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( falcon
   falcon.cpp
   ${SYSTEM_RC}
)

#Link
TARGET_LINK_LIBRARIES(falcon falcon_engine)

#Install
INSTALL( TARGETS falcon
      DESTINATION ${FALCON_BIN_DIR} )
