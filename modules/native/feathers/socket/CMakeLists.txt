####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Socket
####################################################################

find_package( OpenSSL )
if ( OPENSSL_FOUND )
    add_definitions( -DWITH_OPENSSL=1 )
    include_directories( BEFORE ${OPENSSL_INCLUDE_DIR} )
else()
    add_definitions( -DWITH_OPENSSL=0 )
    set( OPENSSL_LIBRARIES "" )
endif()

if(UNIX OR MAC)
   set(SYS_SPECIFIC socket_sys_unix.cpp)

   # On Solaris, we need also socket lib
   if( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
      set(SYS_SPECIFIC_LIBS /lib/libsocket.so)
   endif( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )

else(UNIX OR MAC)
   #We must configure socket_sys_win.cpp.in with the correct location of SDK
   if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles" )
      if(NOT MS_SDK_PATH)
         set( MS_SDK_PATH "$ENV{MS_SDK_PATH}" )
      endif()

      if(NOT MS_SDK_PATH)
         message( FATAL_ERROR "Please, valorize MS_SDK_PATH CMAKE or environment variable with path to SDK" )
      endif()
      include_directories(${MS_SDK_PATH})
   endif()

   set(SYS_SPECIFIC socket_sys_win.cpp socket.rc)
endif(UNIX OR MAC)

# Target
add_library(socket_fm MODULE
   socket.cpp
   socket_ext.cpp
   socket_srv.cpp
   socket_st.cpp
   socket_sys_common.cpp
   ${SYS_SPECIFIC}
)

#Link
target_link_libraries(socket_fm
  falcon_engine
  ${SYS_SPECIFIC_LIBS}
  ${OPENSSL_LIBRARIES}
)
if(WIN32)
   target_link_libraries(socket_fm ws2_32)
endif(WIN32)

falcon_install_module( socket_fm )

set(_project_targets ${_project_targets} socket_fm PARENT_SCOPE)