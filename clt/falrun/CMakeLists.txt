PROJECT(falrun)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
INCLUDE_DIRECTORIES(BEFORE ../../include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )


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
FALCON_INSTALL( falrun bin )
