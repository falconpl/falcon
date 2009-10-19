##################################################
# Falcon Programming Language
#
# Falrun
##################################################

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIRECTORY}
  ${PROJECT_SOURCE_DIR}/include
)


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
INSTALL( TARGETS falrun ${FALCON_INSTALL_DESTINATIONS} )
