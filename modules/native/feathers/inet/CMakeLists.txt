####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Socket
####################################################################

find_package( OpenSSL )
if ( OPENSSL_FOUND )
	option(FALCON_BUILD_INET_WITH_SSL "Build inet modules using SSL library" ON )
else()
	option(FALCON_BUILD_INET_WITH_SSL "Build inet modules using SSL library" OFF )
endif()

if( FALCON_BUILD_INET_WITH_SSL AND NOT OPENSSL_FOUND )
	Message(FATAL "Cannot find SSL libraries in the system") 
endif()

if( FALCON_BUILD_INET_WITH_SSL AND OPENSSL_FOUND )
    add_definitions( -DWITH_OPENSSL=1 )
    include_directories( BEFORE ${OPENSSL_INCLUDE_DIR} )
else()
    add_definitions( -DWITH_OPENSSL=0 )
    set( OPENSSL_LIBRARIES "" )
endif()

if(UNIX OR MAC)
   set(SYS_SPECIFIC inet_mod_posix.cpp)

   # On Solaris, we need also socket lib
   if( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
      set(SYS_SPECIFIC_LIBS /lib/libsocket.so)
   endif( "${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS" )
	
   set(LINK_LIBS "${OPENSSL_LIBRARIES}")
else()
   #We must configure inet_sys_win.cpp.in with the correct location of SDK
   #Not necessary since I fixed it to use the mingw includes.
   # if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles" )
      # if(NOT MS_SDK_PATH)
         # set( MS_SDK_PATH "$ENV{MS_SDK_PATH}" )
      # endif()

      # if(NOT MS_SDK_PATH)
         # message( FATAL_ERROR "Please, valorize MS_SDK_PATH CMAKE or environment variable with path to SDK" )
      # endif()
      # include_directories(${MS_SDK_PATH})
   # endif()

   set(SYS_SPECIFIC inet_mod_win.cpp ../../../../engine/selectmpx.cpp winselectmpx.h winselectmpx.cpp)
   if (NOT MINGW AND NOT FALCON_STATIC_FEATHERS)
      set(SYS_RC inet.rc)
   endif()
   
   set(LINK_LIBS "${OPENSSL_LIBRARIES}" "ws2_32" )
endif()

# Target
set(SOURCES
   inet.cpp
   inet_fm.cpp
   inet_mod.cpp
   
   inet_fm.h
   inet_mod.h
   ${SYS_SPECIFIC}
   ${SYS_RC}
)


FalconDeclareFeather( "inet" "${SOURCES}" "${LINK_LIBS}" )
