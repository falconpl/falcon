PROJECT(falcon)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
INCLUDE_DIRECTORIES(BEFORE ../../include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )


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

#In unix and mac we have to add extra libraries
#Install
FALCON_INSTALL( falcon bin )

