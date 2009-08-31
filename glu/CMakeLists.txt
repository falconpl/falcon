####################################################################
# The Falcon Programming language
#
# CMake configuration file for module SDL_ttf
####################################################################

FALCON_DEFINE_MODULE( SDLGLU sdlglu )

if(COMMAND cmake_policy)
   cmake_policy(SET CMP0003 OLD)
endif(COMMAND cmake_policy)

Find_Package ( SDL REQUIRED )
Find_Package ( OpenGL REQUIRED )
INCLUDE_DIRECTORIES( ${SDL_INCLUDE_DIR} )

# Inclusion settings
INCLUDE_DIRECTORIES(../include)
INCLUDE_DIRECTORIES(.)
INCLUDE_DIRECTORIES(BEFORE "${FALCON_INC_PATH}")
LINK_DIRECTORIES(BEFORE "${FALCON_LIB_PATH}")

# Target
ADD_LIBRARY( ${SDLGLU} MODULE
   glu.cpp
   glu_ext.cpp
   glu_mod.cpp
)

#Link
TARGET_LINK_LIBRARIES( ${SDLGLU} falcon_engine ${SDL_LIBRARY} ${OPENGL_LIBRARY})

SET_TARGET_PROPERTIES( ${SDLGLU}
   PROPERTIES PREFIX "")
#Install
FALCON_LINK_MODULE( ${SDLGLU} )
