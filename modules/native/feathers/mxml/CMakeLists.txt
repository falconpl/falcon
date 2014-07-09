####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Mxml
####################################################################

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}
)

if(WIN32 AND NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
   set(SYS_SPECIFIC  mxml.rc)
else()
	set(SYS_SPECIFIC )
endif()

set(SOURCES
  mxml.cpp
  mxml_fm.cpp
  mxml_attribute.cpp
  mxml_document.cpp
  mxml_error.cpp
  mxml_node.cpp
  mxml_utility.cpp

  mxml_fm.h
  mxml_attribute.h
  mxml_document.h
  mxml_element.h
  mxml_error.h
  mxml_iterator.h
  mxml_node.h
  mxml_utility.h
  mxml.h
  ${SYS_SPECIFIC}
)

FalconDeclareFeather( "mxml" "${SOURCES}" "" )
