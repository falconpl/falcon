####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Funcext
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC funcext.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( funcext MODULE
   funcext.cpp
   funcext_ext.cpp
   ${SYS_RC}
)

#Link
FALCON_LINK_MODULE( funcext )
