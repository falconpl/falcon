##################################################
# Falcon Programming Language
#
# Falcon command line
##################################################

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIRECTORY}
  ${PROJECT_SOURCE_DIR}/include
)


IF(WIN32)
   SET( SYSTEM_RC falcon.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( falcon
   falcon.cpp
   int_mode.cpp
   options.cpp
   ${SYSTEM_RC}
)

#Link
TARGET_LINK_LIBRARIES(falcon falcon_engine)

#Install
INSTALL( TARGETS falcon ${FALCON_INSTALL_DESTINATIONS})
