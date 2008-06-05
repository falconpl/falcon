####################################################################
# The Falcon Programming language
#
# CMake configuration file for module ConfParser
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC confparser.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( confparser MODULE
   confparser.cpp
   confparser_ext.cpp
   confparser_mod.cpp
   confparser_st.cpp
   ${SYS_RC}
)

#Link
FALCON_LINK_MODULE( confparser )
