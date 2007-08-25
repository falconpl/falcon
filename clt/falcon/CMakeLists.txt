PROJECT(falcon)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_BUILD_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../../include)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

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
INSTALL( TARGETS falcon
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/bin )
