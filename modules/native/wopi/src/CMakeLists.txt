######################################################################
# CMake file for falcon WOPI sources
#

#######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Falcon_INCLUDE_DIRS}
)

if(WIN32)
   set( _files_sys 
   )
else()
   set( _files_sys 
   )
endif()

set(_files
   #${_files_sys}
  parthandler.cpp  
  request.cpp
  reply.cpp
  replystream.cpp
  stream_ch.cpp
  uploaded.cpp
  utils.cpp
  wopi.cpp
  client.cpp
  scriptrunner.cpp
  
  classrequest.cpp
  classreply.cpp
  classuploaded.cpp  
  classwopi.cpp
  
  modulewopi.cpp  
)

unset(WOPI_SOURCES)
foreach(item ${_files})
  list(APPEND WOPI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${item})
endforeach(item)

# Sources files the library is built from.
set(WOPI_SOURCES ${WOPI_SOURCES} PARENT_SCOPE)



