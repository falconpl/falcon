####################################################################
# The Falcon Programming language
#
# CMake configuration file for module Compiler
####################################################################
PROJECT(compiler)

# Inclusion settings
INCLUDE_DIRECTORIES(.)

# Target
ADD_LIBRARY( compiler MODULE
compiler.cpp
compiler_ext.cpp
compiler_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES(compiler falcon_engine)
SET_TARGET_PROPERTIES(compiler
   PROPERTIES PREFIX "")
#Install
INSTALL( TARGETS compiler
   DESTINATION $ENV{FALCON_ACTIVE_TREE}/lib/falcon )
