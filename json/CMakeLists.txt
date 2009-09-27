####################################################################
# The Falcon Programming language
#
# CMake configuration file for module json
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC json.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( json_fm MODULE
   json_st.cpp
   json_mod.cpp
   json_ext.cpp
   json.cpp
   ${SYS_RC}
)

#Link
FALCON_LINK_MODULE( json_fm )
