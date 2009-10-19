##################################################
# Falcon Programming Language
#
# Faltest
##################################################

include_directories(
  ${PROJECT_BINARY_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIRECTORY}
  ${PROJECT_SOURCE_DIR}/include
)


IF(WIN32)
   SET(SYS_RC faltest.rc)
   SET(SYS_RC_DLL testsuite.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( faltest
   faltest.cpp
   fal_testsuite.cpp
   scriptdata.cpp
   ${SYS_RC}
)

ADD_LIBRARY( testsuite_fm MODULE
   fal_testsuite.cpp
   ${SYS_RC_DLL}
)

#Link
TARGET_LINK_LIBRARIES(faltest falcon_engine)
TARGET_LINK_LIBRARIES(testsuite_fm falcon_engine)

#faltest embeds the testsuite module
SET_TARGET_PROPERTIES( faltest
   PROPERTIES COMPILE_FLAGS "-DFALCON_EMBED_MODULES")

#while the testsuite has no prefix
IF(APPLE)
SET_TARGET_PROPERTIES( testsuite_fm
   PROPERTIES 
	PREFIX ""
	SUFFIX ".dylib" )
ELSE(APPLE)
SET_TARGET_PROPERTIES( testsuite_fm
   PROPERTIES PREFIX "")
ENDIF(APPLE)

#Install
INSTALL( TARGETS faltest ${FALCON_INSTALL_DESTINATIONS})

INSTALL( TARGETS testsuite_fm
      DESTINATION ${FALCON_MOD_INSTALL_DIR} )
