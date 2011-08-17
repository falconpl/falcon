##################################################
# Falcon Programming Language
#
# Faldisass
##################################################

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIRECTORY}
  ${PROJECT_SOURCE_DIR}/include
)

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
INSTALL( TARGETS faldisass ${FALCON_INSTALL_DESTINATIONS} )
