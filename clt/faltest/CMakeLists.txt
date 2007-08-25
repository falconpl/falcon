PROJECT(faltest)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_BUILD_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../../include)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

IF(WIN32)
   SET(SYS_RC faltest.rc)
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
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/bin )

FALCON_MODULE_INSTALL( testsuite )
