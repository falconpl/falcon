####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Compiler
####################################################################
PROJECT(compiler)

# Inclusion settings
INCLUDE_DIRECTORIES(.)

IF(WIN32)
   SET( SYS_RC compiler.rc )
ENDIF(WIN32)

# Target
ADD_LIBRARY( compiler MODULE
   compiler.cpp
   compiler_ext.cpp
   compiler_mod.cpp

   ${SYS_RC}
)

#Link
TARGET_LINK_LIBRARIES(compiler falcon_engine)
SET_TARGET_PROPERTIES(compiler
   PROPERTIES PREFIX "")
#Install
FALCON_MODULE_INSTALL( compiler )
