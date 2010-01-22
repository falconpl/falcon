##################################################
# Falcon Programming Language
#
# Falpack
##################################################

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIRECTORY}
  ${PROJECT_SOURCE_DIR}/include
)


IF(WIN32)
   SET(SYS_RC falpack.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( falpack
   falpack.cpp
   ${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(falpack falcon_engine)

#Install
INSTALL( TARGETS falpack ${FALCON_INSTALL_DESTINATIONS} )
