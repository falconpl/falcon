####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Process
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(UNIX)
   SET(SYS_SPECIFIC process_sys_unix.cpp)
ELSE(UNIX)
   SET(SYS_SPECIFIC process_sys_win.cpp process.rc)
ENDIF(UNIX)

# Target
ADD_LIBRARY( process_fm MODULE
   process.cpp
   process_ext.cpp
   process_mod.cpp
   process_st.cpp
   ${SYS_SPECIFIC}
)

#Link
FALCON_LINK_MODULE( process_fm )
