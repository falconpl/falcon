######################################################################
# CMake file for falhttpd
#

MESSAGE( "Creating the Falcon micro HTTP Daemon" )

#######################################################################
# Targets
#

# Inclusion settings
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/../feathers/inet
  ${PROJECT_SOURCE_DIR}/../feathers/confparser
  ${Falcon_INCLUDE_DIRS}
)

if(UNIX OR MAC)
	set(SYS_SPECIFIC
		${PROJECT_SOURCE_DIR}/../feathers/inet/inet_mod_posix.cpp
		)
	if(MAC)
		set(SYS_SPECIFIC
			${SYS_SPECIFIC}
			${PROJECT_SOURCE_DIR}/../feathers/inet/inet_mod_posix.cpp
			)
	endif()	
else()
	set(SYS_SPECIFIC
		${PROJECT_SOURCE_DIR}/../feathers/inet/inet_mod_win.cpp
		${PROJECT_SOURCE_DIR}/../feathers/inet/winselectmpx.cpp
		${PROJECT_SOURCE_DIR}/../feathers/inet/winselectmpx.h
		)
endif()

# Sources files the library is built from.
set(SRC_FILES
  falhttpd.cpp
  falhttpd_client.cpp
  falhttpd_dirhandler.cpp
  falhttpd_filehandler.cpp
  falhttpd_options.cpp
  falhttpd_rh.cpp
  falhttpd_scripthandler.cpp
  falhttpd_eh.cpp
  
  falhttpd_loglistener.cpp
  ${PROJECT_SOURCE_DIR}/../feathers/inet/inet_mod.cpp
  ${PROJECT_SOURCE_DIR}/../feathers/inet/inet_ext.cpp
  ${PROJECT_SOURCE_DIR}/../feathers/inet/inet_mod.h
  ${PROJECT_SOURCE_DIR}/../feathers/confparser/confparser_mod.cpp
  ${PROJECT_SOURCE_DIR}/../feathers/confparser/confparser_mod.h
  ${SYS_SPECIFIC}
)
# These are actually not needed by cmake to build. But if omitted they won't be
# listed in the virtual file tree of Visual Studio.
set(HDR_FILES
  falhttpd_client.h
  falhttpd_dirhandler.h
  falhttpd_filehandler.h
  falhttpd.h
  falhttpd_options.h
  falhttpd_rh.h
  falhttpd_scripthandler.h
  falhttpd_eh.h

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
