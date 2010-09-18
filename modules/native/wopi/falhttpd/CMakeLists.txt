######################################################################
# CMake file for fcgi
#

MESSAGE( "Creating the Falcon micro HTTP Daemon" )

#######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${Falcon_INCLUDE_DIRS}
)

# Sources files the library is built from.
set(SRC_FILES
  falhttpd_client.cpp
  falhttpd.cpp
  falhttpd_dirhandler.cpp
  falhttpd_filehandler.cpp
  falhttpd_istream.cpp
  falhttpd_options.cpp
  falhttpd_ostream.cpp
  falhttpd_reply.cpp
  falhttpd_rh.cpp
  falhttpd_scripthandler.cpp
)
# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  falhttpd_client.h
  falhttpd_dirhandler.h
  falhttpd_filehandler.h
  falhttpd.h
  falhttpd_istream.h
  falhttpd_options.h
  falhttpd_ostream.h
  falhttpd_reply.h
  falhttpd_rh.h
  falhttpd_scripthandler.h

  ${WOPI_HEADERS}
)

ADD_EXECUTABLE( falhttpd
   ${SRC_FILES}
   ${WOPI_SOURCES}
   ${HDR_FILES} # optional but useful, see comment above.
)

if(UNIX OR MAC)
   set(SYS_SPECIFIC socket_sys_unix.cpp)

   # On Solaris, we need also socket lib
   if( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
      set(_syslibs 
         /lib/libsocket.so
         /lib/libnsl.so 
      )
   endif()

else()
   set(_syslibs
      ws2_32
   )
endif()


TARGET_LINK_LIBRARIES( falhttpd
   falcon_engine
   ${_syslibs}
   ) 

INSTALL(TARGETS falhttpd
   RUNTIME DESTINATION bin
)
