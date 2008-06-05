##################################################
# Falcon Programming Language
#
# Faltest
##################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET(SYS_RC faltest.rc)
   SET(SYS_RC_DLL testsuite.rc)
ENDIF(WIN32)

# Target
ADD_EXECUTABLE( faltest
   faltest.cpp
   fal_testsuite.cpp
   fteh.cpp
   scriptdata.cpp
   ${SYS_RC}
)

ADD_LIBRARY( testsuite MODULE
   fal_testsuite.cpp
   ${SYS_RC_DLL}
)

#Link
TARGET_LINK_LIBRARIES(faltest falcon_engine)
TARGET_LINK_LIBRARIES(testsuite falcon_engine)

#faltest embeds the testsuite module
SET_TARGET_PROPERTIES( faltest
   PROPERTIES COMPILE_FLAGS "-DFALCON_EMBED_MODULES")

#while the testsuite has no prefix
SET_TARGET_PROPERTIES( testsuite
   PROPERTIES PREFIX "")

#Install
INSTALL( TARGETS faltest
      DESTINATION ${FALCON_BIN_DIR} )

INSTALL( TARGETS testsuite
      DESTINATION ${FALCON_MOD_INSTALL} )
