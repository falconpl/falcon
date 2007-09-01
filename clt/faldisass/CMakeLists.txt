PROJECT(faldisass)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../../include)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

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
FALCON_INSTALL( faldisass bin )
