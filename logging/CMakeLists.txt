####################################################################
# The Falcon Programming language
#
# CMake configuration file for module "logging"
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC logging.rc )
   SET( SYS_SRC logging_srv_win.cpp )
ELSE(WIN32)
   SET( SYS_SRC logging_srv_syslog.cpp )
ENDIF(WIN32)

# Target
ADD_LIBRARY( logging_fm MODULE
   logging.cpp
   logging_ext.cpp
   logging_mod.cpp
   logging_srv.cpp
   logging_st.cpp
   ${SYS_SRC}
   ${SYS_RC}
)

#Link
FALCON_LINK_MODULE( logging_fm )
