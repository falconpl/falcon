PROJECT(falcon_rtl)

# Inclusion settings
INCLUDE_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/include)
INCLUDE_DIRECTORIES(BEFORE ../include)
INCLUDE_DIRECTORIES(.)
LINK_DIRECTORIES(BEFORE $ENV{FALCON_ACTIVE_TREE}/lib )

IF(WIN32)
	SET(SYSTEM_RC  falcon_rtl.rc)
ENDIF(WIN32)

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
   ${SYSTEM_RC}
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
FALCON_MODULE_INSTALL( falcon_rtl )
