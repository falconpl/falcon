####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################
PROJECT(process)

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(UNIX)
   SET(SYS_SPECIFIC process_sys_unix.cpp)
ELSE(UNIX)
   SET(SYS_SPECIFIC process_sys_win.cpp)
ENDIF(UNIX)

# Target
ADD_LIBRARY( process MODULE
process.cpp
process_ext.cpp
process_mod.cpp
${SYS_SPECIFIC}
)

#Link
TARGET_LINK_LIBRARIES(process falcon_engine)
SET_TARGET_PROPERTIES(process
   PROPERTIES PREFIX "")
#Install
INSTALL( TARGETS process
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/lib/falcon )
