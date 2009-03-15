####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Compiler
####################################################################

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC compiler.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( compiler_fm MODULE
   compiler.cpp
   compiler_ext.cpp
   compiler_mod.cpp
   compiler_st.cpp

   ${SYS_RC}
)

#Link
FALCON_LINK_MODULE( compiler_fm )
