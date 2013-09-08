####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Socket
####################################################################

#find_package( OpenSSL )
#if ( OPENSSL_FOUND )
#    add_definitions( -DWITH_OPENSSL=1 )
#    include_directories( BEFORE ${OPENSSL_INCLUDE_DIR} )
#else()
#    add_definitions( -DWITH_OPENSSL=0 )
#    set( OPENSSL_LIBRARIES "" )
#endif()

if(UNIX OR MAC)
   set(SYS_SPECIFIC inet_sys_posix.cpp)

   # On Solaris, we need also socket lib
   if( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
      set(SYS_SPECIFIC_LIBS /lib/libsocket.so)
   endif( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )

else()
   #We must configure inet_sys_win.cpp.in with the correct location of SDK
   if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles" )
      if(NOT MS_SDK_PATH)
         set( MS_SDK_PATH "$ENV{MS_SDK_PATH}" )
      endif()

      if(NOT MS_SDK_PATH)
         message( FATAL_ERROR "Please, valorize MS_SDK_PATH CMAKE or environment variable with path to SDK" )
      endif()
      include_directories(${MS_SDK_PATH})
   endif()

   set(SYS_SPECIFIC inet_sys_win.cpp inet.rc)
endif()

# Target
add_library(inet_fm MODULE
   inet.cpp
   inet_mod.cpp
   inet_ext.cpp
   #${SYS_SPECIFIC}
)

#Link
target_link_libraries(inet_fm
  falcon_engine
  ${SYS_SPECIFIC_LIBS}
  ${OPENSSL_LIBRARIES}
)

if(WIN32)
   target_link_libraries(inet_fm ws2_32)
endif()

falcon_install_module( inet_fm )

set(_project_targets ${_project_targets} inet_fm PARENT_SCOPE)
