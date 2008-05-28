####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################
PROJECT(mxml)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

IF(UNIX)
ELSE(UNIX)
   SET(SYS_SPECIFIC  mxml.rc)
ENDIF(UNIX)


# Target
ADD_LIBRARY( mxml MODULE
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
TARGET_LINK_LIBRARIES(mxml falcon_engine)
SET_TARGET_PROPERTIES(mxml
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( mxml )
