####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################
PROJECT(process)

# Inclusion settings
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE ${TARGET_DIR}/include)
LINK_DIRECTORIES(BEFORE ${TARGET_DIR}/${TARGET_LIB_DIR} )

IF(UNIX)
   SET(SYS_SPECIFIC process_sys_unix.cpp)
ELSE(UNIX)
   SET(SYS_SPECIFIC process_sys_win.cpp process.rc)
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
FALCON_MODULE_INSTALL( process )
