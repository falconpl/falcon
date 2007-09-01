####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################
PROJECT(confparser)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

IF(WIN32)
   SET( SYS_RC confparser.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( confparser MODULE
   confparser.cpp
   confparser_ext.cpp
   confparser_mod.cpp
   ${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(confparser falcon_engine)
SET_TARGET_PROPERTIES(confparser
   PROPERTIES PREFIX "")

#Install
FALCON_MODULE_INSTALL( confparser )
