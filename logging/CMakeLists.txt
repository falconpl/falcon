####################################################################
# The Falcon Programming language
#
# CMake configuration file for module "logging"
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC logging.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( logging_fm MODULE
   logging.cpp
   logging_ext.cpp
   logging_mod.cpp
   logging_st.cpp
   ${SYS_RC}
)

#Link
FALCON_LINK_MODULE( logging_fm )
