PROJECT(falcon_rtl)

# Inclusion settings
INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_BUILD_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../include)
INCLUDE_DIRECTORIES(.)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

# Target

ADD_LIBRARY( falcon_rtl MODULE
   array.cpp
   dict.cpp
   dir_ext.cpp
   envvars.cpp
   falcon_rtl.cpp
   file_ext.cpp
   indirect.cpp
   input.cpp
   inspect.cpp
   itemcopy.cpp
   math.cpp
   messages.cpp
   print.cpp
   random.cpp
   seconds.cpp
   serialize.cpp
   string.cpp
   stringstream_ext.cpp
   time_ext.cpp
   transcode_ext.cpp
)

#Link
TARGET_LINK_LIBRARIES(falcon_rtl falcon_engine)
SET_TARGET_PROPERTIES(falcon_rtl
   PROPERTIES PREFIX "")

#In unix and mac we have to add extra libraries
IF(UNIX OR MAC)
   TARGET_LINK_LIBRARIES(falcon_rtl m)
ENDIF(UNIX OR MAC)

#Install
INSTALL( TARGETS falcon_rtl
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/lib/falcon )
