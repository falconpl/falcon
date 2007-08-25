####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################
PROJECT(confparser)

# Inclusion settings
INCLUDE_DIRECTORIES(.)

# Target
ADD_LIBRARY( confparser MODULE
confparser.cpp
confparser_ext.cpp
confparser_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(confparser falcon_engine)
SET_TARGET_PROPERTIES(confparser
   PROPERTIES PREFIX "")
#Install
INSTALL( TARGETS confparser
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/lib/falcon )
