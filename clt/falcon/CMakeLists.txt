PROJECT(falcon)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_BUILD_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../include)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

# Target
ADD_EXECUTABLE( falcon
   falcon.cpp
)

#Link
TARGET_LINK_LIBRARIES(falcon falcon_engine)

#In unix and mac we have to add extra libraries
#Install
INSTALL( TARGETS falcon
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/bin )
