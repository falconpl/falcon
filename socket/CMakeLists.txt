####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Socket
####################################################################
PROJECT(socket)

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(UNIX)
   SET(SYS_SPECIFIC socket_sys_unix.cpp)
ELSE(UNIX)
   SET(SYS_SPECIFIC socket_sys_win.cpp socket.rc)
ENDIF(UNIX)

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
