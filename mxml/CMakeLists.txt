####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${falcon_INCLUDE_DIRS}
)

if(WIN32)
   set(SYS_SPECIFIC  mxml.rc)
endif()

add_library(mxml_fm MODULE
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

falcon_finalize_module( mxml_fm )
