####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Socket
####################################################################
PROJECT(socket)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/lib )

IF(UNIX OR MAC)
   SET(SYS_SPECIFIC socket_sys_unix.cpp)
ELSE(UNIX OR MAC)
   #We must configure socket_sys_win.cpp.in with the correct location of SDK
   IF( "${CMAKE_GENERATOR}" STREQUAL "MinGW Makefiles" )
      IF( "${MS_SDK_PATH}" STREQUAL "" )
         SET( MS_SDK_PATH "$ENV{MS_SDK_PATH}" )
      ENDIF( "${MS_SDK_PATH}" STREQUAL "" )
      
      IF( "${MS_SDK_PATH}" STREQUAL "" )
         MESSAGE( FATAL_ERROR "Please, valorize MS_SDK_PATH CMAKE or environment variable with path to SDK" )
      ENDIF( "${MS_SDK_PATH}" STREQUAL "" )
      INCLUDE_DIRECTORIES(${MS_SDK_PATH})
   ENDIF("${CMAKE_GENERATOR}" STREQUAL "MinGW Makefiles" )
   
   SET(SYS_SPECIFIC socket_sys_win.cpp socket.rc)
ENDIF(UNIX OR MAC)

# Target
ADD_LIBRARY( socket MODULE
   socket.cpp
   socket_ext.cpp
   socket_srv.cpp
   ${SYS_SPECIFIC}
)

#Link
TARGET_LINK_LIBRARIES(socket falcon_engine)
IF(WIN32)
   TARGET_LINK_LIBRARIES(socket ws2_32)
ENDIF(WIN32)

SET_TARGET_PROPERTIES(socket
   PROPERTIES PREFIX "")

#Install
FALCON_MODULE_INSTALL( socket )
