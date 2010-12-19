####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################

FALCON_DEFINE_MODULE( SDLGLEXT glext )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)

Find_Package ( SDL REQUIRED )
Find_Package ( OpenGL REQUIRED )

# Inclusion settings
INCLUDE_DIRECTORIES(
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Target
ADD_LIBRARY( ${SDLGLEXT} MODULE
   glext.cpp
   glext_ext.cpp
   glext_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES( ${SDLGLEXT} falcon_engine ${SDL_LIBRARY} ${OPENGL_LIBRARY})

SET_TARGET_PROPERTIES( ${SDLGLEXT}
   PROPERTIES PREFIX "")
#Install
FALCON_INSTALL_MODULE2( ${SDLGLEXT} sdl )
