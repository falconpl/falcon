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
   SET(SYS_SPECIFIC socket_sys_win.cpp)
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
SET_TARGET_PROPERTIES(socket
   PROPERTIES PREFIX "")
#Install
INSTALL( TARGETS socket
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/lib/falcon )
