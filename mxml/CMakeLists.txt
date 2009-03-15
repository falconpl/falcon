####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(UNIX)
ELSE(UNIX)
   SET(SYS_SPECIFIC  mxml.rc)
ENDIF(UNIX)


# Target
ADD_LIBRARY( mxml_fm MODULE
   mxml.cpp
   mxml_ext.cpp
   mxml_mod.cpp
   mxml_attribute.cpp
   mxml_document.cpp
   mxml_error.cpp
   mxml_node.cpp
   mxml_utility.cpp
   mxml_st.cpp
   ${SYS_SPECIFIC}
)

#Link
FALCON_LINK_MODULE( mxml_fm )
