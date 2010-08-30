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
        sharedmem_win.cpp
   )
else()
   set( _files_sys 
        sharedmem_posix.cpp
   )
endif()

set(_files
   ${_files_sys}
  error_ext.cpp
  file_sm.cpp
  mem_sm.cpp
  parthandler.cpp
  reply.cpp
  reply_ext.cpp
  replystream.cpp
  request.cpp
  request_ext.cpp
  session_manager.cpp
  utils.cpp
  uploaded_ext.cpp
  wopi.cpp
  wopi_ext.cpp
)

unset(WOPI_SOURCES)
foreach(item ${_files})
  list(APPEND WOPI_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/${item})
endforeach(item)

# Sources files the library is built from.
set(WOPI_SOURCES ${WOPI_SOURCES} PARENT_SCOPE)



